#!/usr/bin/env python3
"""Inter-host PP prototype: DS4-Flash split into two pipeline stages
(layers 0-21 + embed | layers 22-42 + head), activations over real TCP
with an injectable WAN RTT, ROUTED EXPERTS SERVED BY THE EXL3 PACK
(Exl3BaseTier / exl3_mgemm); attention, dense, shared expert and gate run
on the official reference implementation (tilelang kernels).

This is the mechanics/physics prototype for the "serve-from-RAM + PP
between hosts" design (EXL3_BASE_TIER_FEASIBILITY.md, Inter-host PP
section) — NOT a performance build (python layer loop, no cudagraphs,
no speculation): per-token time decomposes into stage compute + link,
and the link term is what we are validating.

Usage:
  # stage 1 first (listens):
  CUDA_VISIBLE_DEVICES=2 python3 pp_proto.py --stage 1 --port 29777 --rtt-ms 46.6
  # then stage 0 (connects, drives, prints text + timing):
  CUDA_VISIBLE_DEVICES=0 python3 pp_proto.py --stage 0 --peer 127.0.0.1:29777 \
      --rtt-ms 46.6 --new-tokens 48
"""
import argparse
import importlib.util
import json
import os
import pickle
import socket
import struct as pystruct
import sys
import time

import torch

AB_DIR = "/root/workspace/exl3-ab"
sys.path.insert(0, AB_DIR)
import capture_hidden as CH  # patched official impl (sparse_attn, helpers)

M = CH.M
PACK = "/root/workspace/moet-serve/exl3-packs-ds4"
CKPT = CH.CKPT
SPLIT = 22
N_LAYERS = 43

spec = importlib.util.spec_from_file_location(
    "moe_w2_exl3",
    "/root/workspace/vllm-wt-exl3-base/vllm/model_executor/layers/"
    "quantization/utils/moe_w2_exl3.py")
X3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X3)


# ------------------------------------------------------------ tcp helpers

def send_msg(sock, obj, one_way_s):
    data = pickle.dumps(obj, protocol=4)
    if one_way_s > 0:
        time.sleep(one_way_s)
    sock.sendall(pystruct.pack("<Q", len(data)) + data)


def recv_msg(sock):
    hdr = b""
    while len(hdr) < 8:
        c = sock.recv(8 - len(hdr))
        if not c:
            raise ConnectionError("peer closed")
        hdr += c
    (n,) = pystruct.unpack("<Q", hdr)
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(min(1 << 20, n - len(buf)))
        if not c:
            raise ConnectionError("peer closed")
        buf += c
    return pickle.loads(bytes(buf))


# ------------------------------------------------------- model per stage

def build_stage(stage, dev):
    cfg = json.load(open(os.path.join(CKPT, "inference", "config.json")))
    margs = M.ModelArgs(**cfg, max_batch_size=1, max_seq_len=4096)
    torch.set_default_dtype(torch.bfloat16)
    model = M.Transformer(margs)
    loaded, _, missing = CH.load_checkpoint(model)
    assert not missing, missing
    lo, hi = (0, SPLIT) if stage == 0 else (SPLIT, N_LAYERS)

    # EXL3 tier replaces the routed-expert loop; drop the FP4 expert
    # modules BEFORE moving layers to GPU (they would not fit).
    tier = X3.Exl3BaseTier(PACK, layers=range(lo, hi), device=dev)
    for li in range(lo, hi):
        ffn = model.layers[li].ffn
        ffn.experts = torch.nn.ModuleList([None] * margs.n_routed_experts)
        ffn._exl3_li = li

    def moe_forward(self, x, input_ids):
        shape = x.size()
        xf = x.view(-1, self.dim)
        weights, indices = self.gate(xf, input_ids.flatten())
        y = tier.forward_topk(self._exl3_li, xf, indices.long(),
                              weights.float())
        y = y + self.shared_experts(xf).float()
        return y.type_as(x).view(shape)

    import types as _t
    for li in range(lo, hi):
        model.layers[li].ffn.forward = _t.MethodType(moe_forward,
                                                     model.layers[li].ffn)

    t0 = time.time()
    if stage == 0:
        model.embed.to(dev)
    for li in range(lo, hi):
        model.layers[li].to(dev)
    if stage == 1:
        model.norm.to(dev)
        model.head.to(dev)
        for p in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            getattr(model, p).data = getattr(model, p).data.to(dev)
    print(f"[stage{stage}] layers {lo}-{hi-1} on {dev} "
          f"({time.time()-t0:.0f}s; exl3 {tier.total_bytes()/2**30:.1f} GiB)",
          flush=True)
    return model, margs, (lo, hi)


def run_layers(model, h, ids, start_pos, lo, hi, dev):
    with torch.device(dev):
        for li in range(lo, hi):
            h = model.layers[li](h, start_pos, ids)
    return h


# --------------------------------------------------------------- stages

@torch.inference_mode()
def stage0(args):
    dev = "cuda:0"
    model, margs, (lo, hi) = build_stage(0, dev)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT)
    prompt = args.prompt
    ids = tok.encode(prompt, add_special_tokens=False)
    print(f"[stage0] prompt: {len(ids)} tokens", flush=True)

    host, port = args.peer.split(":")
    sock = socket.create_connection((host, int(port)))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    ow = args.rtt_ms / 2000.0

    def fwd(token_ids, start_pos):
        t = torch.tensor([token_ids], dtype=torch.long, device=dev)
        with torch.device(dev):
            h = model.embed(t)
            h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        h = run_layers(model, h, t, start_pos, lo, hi, dev)
        torch.cuda.synchronize()
        return h

    def pack_h(h):
        # bf16 has no numpy dtype: ship the raw bits (exact roundtrip)
        return h.contiguous().view(torch.uint16).cpu().numpy()

    # prefill
    t0 = time.time()
    h = fwd(ids, 0)
    send_msg(sock, {"h": pack_h(h), "ids": ids, "start_pos": 0}, ow)
    r = recv_msg(sock)
    cur = r["token"]
    prefill_s = time.time() - t0
    out_ids = [cur]

    # decode
    lat, comp = [], []
    for i in range(args.new_tokens - 1):
        t0 = time.time()
        h = fwd([cur], len(ids) + i)
        tc = time.time() - t0
        send_msg(sock, {"h": pack_h(h), "ids": [cur],
                        "start_pos": len(ids) + i}, ow)
        r = recv_msg(sock)
        cur = r["token"]
        out_ids.append(cur)
        lat.append(time.time() - t0)
        comp.append(tc + r["comp_s"])
    send_msg(sock, {"stop": True}, ow)

    text = tok.decode(out_ids)
    n = len(lat)
    print(f"\n[stage0] prefill {prefill_s:.1f}s; decode {n} tok:"
          f" mean {sum(lat)/n*1000:.0f} ms/tok = {n/sum(lat):.2f} tok/s"
          f" (compute {sum(comp)/n*1000:.0f} ms, link+ser "
          f"{(sum(lat)-sum(comp))/n*1000:.0f} ms; injected RTT {args.rtt_ms} ms)")
    print(f"[stage0] TEXT: {text!r}")


@torch.inference_mode()
def stage1(args):
    dev = "cuda:0"
    model, margs, (lo, hi) = build_stage(1, dev)
    srv = socket.create_server(("0.0.0.0", args.port))
    print(f"[stage1] listening :{args.port}", flush=True)
    sock, addr = srv.accept()
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[stage1] peer {addr}", flush=True)
    ow = args.rtt_ms / 2000.0

    while True:
        msg = recv_msg(sock)
        if msg.get("stop"):
            break
        t0 = time.time()
        h = torch.from_numpy(msg["h"]).view(torch.bfloat16).to(dev)
        t = torch.tensor([msg["ids"]], dtype=torch.long, device=dev)
        h = run_layers(model, h, t, msg["start_pos"], lo, hi, dev)
        with torch.device(dev):
            logits = model.head(h, model.hc_head_fn, model.hc_head_scale,
                                model.hc_head_base, model.norm)
        token = int(logits[:, -1].argmax(-1)[0]) if logits.dim() == 3 \
            else int(logits.argmax(-1)[-1])
        torch.cuda.synchronize()
        send_msg(sock, {"token": token, "comp_s": time.time() - t0}, ow)
    print("[stage1] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--peer", default="127.0.0.1:29777")
    ap.add_argument("--port", type=int, default=29777)
    ap.add_argument("--rtt-ms", type=float, default=0.0)
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--prompt", default="The capital of Poland is Warsaw. "
                    "The boiling point of water at sea level is")
    args = ap.parse_args()
    (stage0 if args.stage == 0 else stage1)(args)


if __name__ == "__main__":
    main()

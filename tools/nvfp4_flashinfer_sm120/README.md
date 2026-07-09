# NVFP4 KV cache for FlashInfer sparse-MLA SM120 (`--kv-cache-dtype nvfp4`)

Packed NVFP4 KV cache for the DeepSeek-V3.2 / GLM sparse-MLA path on SM120
(RTX PRO 6000 Blackwell / RTX 5090): **352 B/token instead of 656 B
(fp8_ds_mla) — 1.86× less KV traffic, ~1.72× more KV pool tokens end to end**
(the DSA indexer cache stays fp8).

## Layout (V1, 352 B/token, flat addressing `idx * 352`)

| bytes | contents |
|---|---|
| `[0:256)` | 512 × E2M1 nibbles (even dim in the low nibble) |
| `[256:288)` | 32 × E4M3 block-16 scales; `dequant = e4m3 × 2⁻⁶` |
| `[288:352)` | 64 × FP8 E4M3 rope, scale 1.0 |

The global scale is a fixed 2⁻⁶ (writes are incremental, so a per-page
dynamic global scale is impossible; measured on real GLM-5.2 latents the
fixed constant is within 0.0002 rel-RMS of a dynamic per-tensor one, with
~10× amax headroom).

## How the kernels read it

The vLLM side (this branch) writes the packed layout
(`csrc/nvfp4_ds_mla/concat_and_cache_nvfp4_ds_mla.cu`) and passes
`kv_scale_format="nvfp4_b16"` to FlashInfer. The FlashInfer side is patched
by `patch_flashinfer.py` (run inside the serving image against the installed
`flashinfer` package; the JIT recompiles from the patched sources):

- new `ModelType::GLM_NSA_NVFP4` (=3) with `KV_GMEM_STRIDE=352`,
- decode/prefill bulk-copy the packed 288 B (+64 B rope) to the **tail** of
  the existing 528 B (128 B) smem slots,
- before QK, the math warps expand in place (PRMT LUT E2M1→E4M3, rescale to
  a per-tile-128 max scale, rope E4M3→BF16) into the exact GLM_NSA smem
  layout — the whole FP8-MMA/softmax/XV pipeline runs unchanged.

The tile-128 requant adds 0.0949 → 0.0967 rel-RMS on real latents — the same
error composition as the fake-quant + fp8_ds_mla-write path that was
validated end-to-end (output divergence at the server nondeterminism floor,
needle 9k–324k all pass).

## Measured (GLM-5.2-NVFP4 744B, 8× RTX PRO 6000, TP8, MTP)

- KV pool @0.90 util: DCP=1: 721k tokens · DCP=2: 1.44M · DCP=4: 2.97M ·
  DCP=8: 5.75M (fp8_ds_mla @0.92, DCP=2 was 859k).
- Decode at 480k context: DCP=1: 61 tok/s · DCP=2: 52 · DCP=4: 42 · DCP=8: 33
  (300 W power cap; parity with fp8_ds_mla at equal settings — sparse decode
  reads only top-2048 tokens, so the win is capacity, not decode tok/s).
- Quality: greedy battery + needle at the server-nondeterminism noise floor.
- Microbench (isolated KV gather, RTX 5090): 1.86× tokens/s vs 656 B.

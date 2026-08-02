# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import multiprocessing
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import triton

import vllm.envs as vllm_envs
from vllm.model_executor.layers.quantization.utils import (
    moe_w2_cubit,
    moe_w2_delta,
    moe_w2_gate,
    moe_w2_planes,
    moe_w2_planes_cache,
    moe_w2_store,
)


@pytest.fixture
def pack_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_MOE_W2_STORE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_MOE_W2_CKPT_ID", "checkpoint-a")
    monkeypatch.setenv("VLLM_MOE_W2_ZERO_MODE", "auto")
    moe_w2_planes_cache._ckpt_id.cache_clear()
    yield tmp_path
    moe_w2_planes_cache._ckpt_id.cache_clear()


def _parts(experts=2, slot_bytes=16):
    left = torch.arange(experts * 8, dtype=torch.uint8).reshape(experts, 8)
    right = torch.arange(
        experts * (slot_bytes - 8), dtype=torch.uint8
    ).reshape(experts, slot_bytes - 8)
    return left, right


def _built_store(pack_env):
    store = moe_w2_store.MmapPackStore(
        str(pack_env), "base", n_layers=2, n_experts=2, slot_bytes=16
    )
    store.add_layer(0, _parts())
    return store


def _writer_process(root, layer_key, value):
    os.environ["VLLM_MOE_W2_STORE_DIR"] = root
    os.environ["VLLM_MOE_W2_CKPT_ID"] = "checkpoint-concurrent"
    os.environ["VLLM_MOE_W2_ZERO_MODE"] = "auto"
    moe_w2_planes_cache._ckpt_id.cache_clear()
    store = moe_w2_store.MmapPackStore(
        root, "base", n_layers=2, n_experts=2, slot_bytes=16)
    parts = (
        torch.full((2, 8), value, dtype=torch.uint8),
        torch.full((2, 8), value + 1, dtype=torch.uint8),
    )
    store.add_layer(layer_key, parts)
    store.release()


def test_loader_probe_requires_pack_bytes(pack_env):
    store = _built_store(pack_env)
    assert moe_w2_store.pack_has_layer("base", 0, 2, 2, 16)
    path = store.path
    store.release()

    os.unlink(path)
    assert not moe_w2_store.pack_has_layer("base", 0, 2, 2, 16)

    rebuilt = moe_w2_store.MmapPackStore(
        str(pack_env), "base", n_layers=2, n_experts=2, slot_bytes=16
    )
    assert 0 not in rebuilt
    assert os.path.getsize(rebuilt.path) == 2 * 2 * 4096
    rebuilt.release()


def test_loader_probe_rejects_truncated_pack(pack_env):
    store = _built_store(pack_env)
    path = store.path
    store.release()
    with open(path, "r+b") as f:
        f.truncate(4096)

    assert not moe_w2_store.pack_has_layer("base", 0, 2, 2, 16)
    rebuilt = moe_w2_store.MmapPackStore(
        str(pack_env), "base", n_layers=2, n_experts=2, slot_bytes=16
    )
    assert len(rebuilt) == 0
    rebuilt.release()


def test_identity_and_quantizer_mode_change_namespace(pack_env, monkeypatch):
    store = _built_store(pack_env)
    old_path = store.path
    store.release()

    monkeypatch.setenv("VLLM_MOE_W2_ZERO_MODE", "alt")
    assert not moe_w2_store.pack_has_layer("base", 0, 2, 2, 16)
    alt = moe_w2_store.MmapPackStore(
        str(pack_env), "base", n_layers=2, n_experts=2, slot_bytes=16
    )
    assert alt.path != old_path
    alt.release()

    monkeypatch.setenv("VLLM_MOE_W2_CKPT_ID", "checkpoint-b")
    moe_w2_planes_cache._ckpt_id.cache_clear()
    assert not moe_w2_store.pack_has_layer("base", 0, 2, 2, 16)


def test_planes_cache_shutdown_clears_checkpoint_identity(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_W2_CKPT_ID", "checkpoint-a")
    moe_w2_planes_cache._ckpt_id.cache_clear()
    assert moe_w2_planes_cache._ckpt_id() == "checkpoint-a"
    monkeypatch.setenv("VLLM_MOE_W2_CKPT_ID", "checkpoint-b")
    assert moe_w2_planes_cache._ckpt_id() == "checkpoint-a"
    moe_w2_planes_cache.shutdown()
    assert moe_w2_planes_cache._ckpt_id() == "checkpoint-b"


def test_invalid_zero_mode_fails_at_startup(pack_env, monkeypatch):
    monkeypatch.setenv("VLLM_MOE_W2_ZERO_MODE", "typo")
    with pytest.raises(ValueError, match="ZERO_MODE"):
        moe_w2_store.MmapPackStore(
            str(pack_env), "base", n_layers=2, n_experts=2, slot_bytes=16
        )


def test_manifest_geometry_and_layer_range_are_strict(pack_env):
    store = _built_store(pack_env)
    sidecar = store._sidecar_path
    store.release()
    with open(sidecar) as f:
        meta = json.load(f)
    meta["layers"] = [0, 99]
    with open(sidecar, "w") as f:
        json.dump(meta, f)

    assert not moe_w2_store.pack_has_layer("base", 0, 2, 2, 16)


def test_concurrent_pack_writers_merge_layer_manifest(tmp_path, monkeypatch):
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(target=_writer_process, args=(str(tmp_path), 0, 10)),
        ctx.Process(target=_writer_process, args=(str(tmp_path), 1, 20)),
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0
    monkeypatch.setenv("VLLM_MOE_W2_STORE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_MOE_W2_CKPT_ID", "checkpoint-concurrent")
    monkeypatch.setenv("VLLM_MOE_W2_ZERO_MODE", "auto")
    moe_w2_planes_cache._ckpt_id.cache_clear()
    store = moe_w2_store.MmapPackStore(
        str(tmp_path), "base", n_layers=2, n_experts=2, slot_bytes=16)
    assert set(store._present) == {0, 1}
    rows = store.rows_for([(0, 0), (1, 0)])
    assert torch.equal(rows[0].cpu(), torch.tensor(
        [10] * 8 + [11] * 8, dtype=torch.uint8))
    assert torch.equal(rows[1].cpu(), torch.tensor(
        [20] * 8 + [21] * 8, dtype=torch.uint8))
    store.release()


def test_rectangular_fp8_block_shape_matches_direct_dequant():
    weight = torch.linspace(-2, 2, 64 * 128, dtype=torch.float32)
    weight = weight.reshape(64, 128).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.5, 1.0], [2.0, 4.0]], dtype=torch.float32)
    got = moe_w2_planes.fp8_block_to_codes_scales(
        weight, scales, block_shape=(32, 64)
    )
    expanded = scales.repeat_interleave(32, 0).repeat_interleave(64, 1)
    expected = moe_w2_planes._f64_to_codes_scales(
        weight.double() * expanded.double()
    )
    assert torch.equal(got[0], expected[0])
    assert torch.equal(got[1], expected[1])


def test_fp8_block_scale_shape_is_validated():
    weight = torch.ones((64, 128), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="scale shape"):
        moe_w2_planes.fp8_block_to_codes_scales(
            weight, torch.ones((1, 1)), block_shape=(32, 64)
        )


def _contract_layer(**overrides):
    values = dict(
        activation="silu",
        swiglu_limit=10.0,
        swiglu_alpha=None,
        swiglu_beta=None,
        moe_config=SimpleNamespace(has_bias=False),
        expert_map=None,
        apply_router_weight_on_input=False,
        layer_name="model.layers.0.mlp.experts",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_w2_layer_contract_preserves_ds4_clamp():
    contract = moe_w2_cubit._layer_contract(_contract_layer())
    assert contract["activation"] == "silu"
    assert contract["swiglu_limit"] == 10.0
    assert contract["swiglu_alpha"] == 1.0
    assert contract["swiglu_beta"] == 0.0


def test_w2_layer_contract_allows_diagnostic_unclamped_ab(monkeypatch):
    monkeypatch.setattr(moe_w2_cubit, "_SWIGLU_CLAMP", False)
    contract = moe_w2_cubit._layer_contract(_contract_layer())
    assert contract["swiglu_limit"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w2_clamp_matches_native_deepgemm_precision():
    torch.manual_seed(0)
    x = (
        torch.randn(17, 512, device="cuda", dtype=torch.bfloat16) * 12
    ).contiguous()
    out = torch.empty(17, 256, device="cuda", dtype=torch.bfloat16)
    moe_w2_cubit._silu_and_mul_clamp_fp32(out, x, 10.0)

    gate = torch.minimum(x[:, :256].float(), torch.tensor(10.0, device="cuda"))
    up = torch.clamp(x[:, 256:].float(), -10.0, 10.0)
    ref = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=0, atol=4e-3)


@pytest.mark.parametrize(
    "dtype,use_ubatching,expert_parallel,match",
    [
        (torch.float16, False, False, "require model dtype"),
        (torch.bfloat16, True, False, "ubatching"),
        (torch.bfloat16, False, True, "expert parallelism"),
    ],
)
def test_w2_config_contract_rejects_unsafe_runtime_combinations(
        dtype, use_ubatching, expert_parallel, match):
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=dtype),
        use_v2_model_runner=False,
        parallel_config=SimpleNamespace(
            use_ubatching=use_ubatching,
            ubatch_size=1 if use_ubatching else 0,
            enable_expert_parallel=expert_parallel,
            enable_eplb=False,
        ),
    )
    with mock.patch(
            "vllm.config.get_current_vllm_config", return_value=config):
        with pytest.raises(ValueError, match=match):
            moe_w2_cubit._layer_contract(_contract_layer())


def _v2_contract_config(speculative_config=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        use_v2_model_runner=True,
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(
            use_ubatching=False,
            ubatch_size=0,
            enable_expert_parallel=False,
            enable_eplb=False,
        ),
    )


def test_w2_v2_allows_full_resident_gate_disabled(monkeypatch):
    monkeypatch.setattr(moe_w2_delta, "base_enabled", lambda: False)
    monkeypatch.setattr(moe_w2_gate, "enabled", lambda: False)
    with mock.patch(
        "vllm.config.get_current_vllm_config",
        return_value=_v2_contract_config(),
    ):
        contract = moe_w2_cubit._layer_contract(_contract_layer())
    assert contract["activation"] == "silu"


def _dspark_config(n_mtp_layers=3):
    return SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(n_mtp_layers=n_mtp_layers)
        ),
    )


def test_w2_mtp_layer_selection_is_explicit(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_W2", "1")
    monkeypatch.setattr(moe_w2_cubit, "_cutoff_cache", 43)
    monkeypatch.delenv("VLLM_MOE_W2_MTP_LAYERS", raising=False)
    assert moe_w2_cubit.is_w2_layer("model.layers.42.ffn.experts")
    assert not moe_w2_cubit.is_w2_layer("model.layers.43.ffn.experts")

    monkeypatch.setenv("VLLM_MOE_W2_MTP_LAYERS", "3")
    assert moe_w2_cubit.is_w2_layer("model.layers.43.ffn.experts")
    assert moe_w2_cubit.is_w2_layer("model.layers.45.ffn.experts")
    assert not moe_w2_cubit.is_w2_layer("model.layers.46.ffn.experts")


def test_w2_mtp_layer_count_rejects_negative_values(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_W2_MTP_LAYERS", "-1")
    with pytest.raises(ValueError, match="must be non-negative"):
        moe_w2_cubit._mtp_layer_count()


def test_w2_mtp_layers_do_not_extend_the_target_delta_tier(monkeypatch):
    monkeypatch.setattr(moe_w2_cubit, "_cutoff_cache", 43)
    layer = SimpleNamespace(layer_name="model.layers.43.ffn.experts")
    with mock.patch.object(
        moe_w2_delta,
        "get_tier",
        side_effect=AssertionError("MTP must not create or index target tier"),
    ):
        assert moe_w2_cubit._fp4_tier_for_build(
            256, torch.device("cpu"), 1024, 512, layer
        ) is None


@pytest.mark.parametrize(
    "use_v2,spec_config,mtp_layers,match",
    [
        (False, _dspark_config(), 3, "requires Model Runner V2"),
        (True, None, 3, "only with DSpark"),
        (
            True,
            SimpleNamespace(method="eagle"),
            3,
            "only with DSpark",
        ),
        (
            True,
            _dspark_config(),
            2,
            "must match DSpark's resolved draft layer count",
        ),
    ],
)
def test_w2_mtp_rejects_unsupported_runtime_configs(
    monkeypatch, use_v2, spec_config, mtp_layers, match
):
    monkeypatch.setenv("VLLM_MOE_W2_MTP_LAYERS", str(mtp_layers))
    monkeypatch.setattr(moe_w2_delta, "base_enabled", lambda: False)
    monkeypatch.setattr(moe_w2_gate, "enabled", lambda: False)
    config = _v2_contract_config(spec_config)
    config.use_v2_model_runner = use_v2
    with mock.patch(
        "vllm.config.get_current_vllm_config", return_value=config
    ):
        with pytest.raises(ValueError, match=match):
            moe_w2_cubit._layer_contract(_contract_layer())


def test_w2_mtp_missing_metadata_defers_until_dspark_construction(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_W2_MTP_LAYERS", "3")
    monkeypatch.setattr(moe_w2_delta, "base_enabled", lambda: False)
    monkeypatch.setattr(moe_w2_gate, "enabled", lambda: False)
    spec_config = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace()),
    )
    with mock.patch(
        "vllm.config.get_current_vllm_config",
        return_value=_v2_contract_config(spec_config),
    ):
        moe_w2_cubit._layer_contract(_contract_layer())


def test_w2_mtp_deferred_validation_uses_resolved_dspark_count(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_W2_MTP_LAYERS", "3")
    moe_w2_cubit.validate_mtp_layer_count(3)
    with pytest.raises(ValueError, match="resolved draft layer count"):
        moe_w2_cubit.validate_mtp_layer_count(2)


def test_w2_mtp_accepts_complete_dspark_draft(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_W2_MTP_LAYERS", "3")
    monkeypatch.setattr(moe_w2_delta, "base_enabled", lambda: False)
    monkeypatch.setattr(moe_w2_gate, "enabled", lambda: False)
    with mock.patch(
        "vllm.config.get_current_vllm_config",
        return_value=_v2_contract_config(_dspark_config()),
    ):
        contract = moe_w2_cubit._layer_contract(_contract_layer())
    assert contract["activation"] == "silu"


@pytest.mark.parametrize(
    "base_enabled,gate_enabled,match",
    [
        (True, False, "base cache/replay"),
        (False, True, "confidence-gate"),
    ],
)
def test_w2_v2_rejects_replay_dependent_modes(
    monkeypatch, base_enabled, gate_enabled, match
):
    monkeypatch.setattr(moe_w2_delta, "base_enabled", lambda: base_enabled)
    monkeypatch.setattr(moe_w2_gate, "enabled", lambda: gate_enabled)
    with mock.patch(
        "vllm.config.get_current_vllm_config",
        return_value=_v2_contract_config(),
    ):
        with pytest.raises(ValueError, match=match):
            moe_w2_cubit._layer_contract(_contract_layer())


def test_w2_v2_lifecycle_is_rank_local_and_ordered(monkeypatch):
    calls = []

    class Tier:
        def __init__(self, name):
            self.name = name

        def wait_manager_idle(self):
            calls.append((self.name, "wait"))

        def step_end(self):
            calls.append((self.name, "end"))

        def wake(self):
            calls.append((self.name, "wake"))

    monkeypatch.setattr(moe_w2_delta, "_BASE_TIER", Tier("base"))
    monkeypatch.setattr(moe_w2_delta, "_TIER", Tier("delta"))
    moe_w2_delta.v2_runner_before_forward()
    moe_w2_delta.v2_runner_step_end()
    moe_w2_delta.wake_all()
    assert calls == [
        ("base", "wait"),
        ("delta", "wait"),
        ("base", "end"),
        ("delta", "end"),
        ("base", "wake"),
        ("delta", "wake"),
    ]


def test_strict_replay_converges_or_fails_closed():
    if moe_w2_delta._REPLAY_MODE != "strict":
        pytest.skip("environment explicitly selected approximate replay")
    assert moe_w2_delta.fp_continue(1, 100)
    assert not moe_w2_delta.fp_continue(moe_w2_delta._FP_MAX, 100)
    with pytest.raises(RuntimeError, match="did not converge"):
        moe_w2_delta.fp_validate_complete(100)
    moe_w2_delta.fp_validate_complete(0)


def test_strict_replay_allows_deep_fixed_point(monkeypatch):
    monkeypatch.setattr(moe_w2_delta, "_REPLAY_MODE", "strict")
    monkeypatch.setattr(moe_w2_delta, "_FP_MAX", 32)
    assert moe_w2_delta.fp_continue(8, 1)
    assert moe_w2_delta.fp_continue(31, 1)
    assert not moe_w2_delta.fp_continue(32, 1)


def test_gate_replay_base_misses_follow_replay_contract(monkeypatch):
    monkeypatch.setattr(moe_w2_delta, "_BASE_MISS_TOL", 0)
    monkeypatch.setattr(moe_w2_delta, "_BASE_MISS_TOL_FILE", "")
    monkeypatch.setattr(moe_w2_delta, "_REPLAY_MODE", "strict")
    moe_w2_delta.gate_validate_base_clean(0)
    with pytest.raises(RuntimeError, match="gate replay introduced"):
        moe_w2_delta.gate_validate_base_clean(1)

    monkeypatch.setattr(moe_w2_delta, "_REPLAY_MODE", "approximate")
    moe_w2_delta.gate_validate_base_clean(1)


def test_mandatory_promotion_failure_is_fail_closed():
    from vllm.v1.worker.gpu_model_runner import _moe_w2_promote_consensus

    def fail(**_kwargs):
        raise IOError("injected pack fault")

    tier = SimpleNamespace(
        force_promote=fail,
        dev=torch.device("cpu"),
    )
    group = SimpleNamespace(world_size=1)
    with pytest.raises(RuntimeError, match="refusing to replay"):
        _moe_w2_promote_consensus(
            tier, group, 1, pin=True, where="unit test")


def test_cubit_shutdown_resets_model_owned_globals():
    moe_w2_cubit._LAYERS[99] = {"sentinel": True}
    moe_w2_cubit._WS["sentinel"] = torch.tensor(1)
    moe_w2_cubit._n_created = 7
    moe_w2_cubit._cutoff_cache = 43
    moe_w2_cubit.shutdown()
    assert not moe_w2_cubit._LAYERS
    assert not moe_w2_cubit._WS
    assert moe_w2_cubit._n_created == 0
    assert moe_w2_cubit._cutoff_cache is None


def test_moet_extension_env_is_validated_and_hashed(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_W2_TEST_FACTOR", "sentinel")
    vllm_envs.validate_environ(hard_fail=True)
    assert vllm_envs.compile_factors()[
        "VLLM_MOE_W2_TEST_FACTOR"] == "sentinel"


def test_known_nan_dense_fp8_mode_is_hard_disabled(monkeypatch):
    from vllm.model_executor.layers.quantization.modelopt import (
        _maybe_dense_fp8_method,
    )
    monkeypatch.setenv("VLLM_MOE_W2_DENSE_FP8", "attn")
    with pytest.raises(ValueError, match="unsafe bring-up"):
        _maybe_dense_fp8_method("model.layers.0.self_attn", object())


@pytest.mark.parametrize(
    "override,match",
    [
        ({"activation": "gelu"}, "only packed SILU"),
        ({"swiglu_alpha": 1.702}, "alpha/beta"),
        ({"moe_config": SimpleNamespace(has_bias=True)}, "bias"),
        ({"expert_map": torch.tensor([0])}, "expert parallel"),
        ({"apply_router_weight_on_input": True}, "router weight"),
    ],
)
def test_w2_layer_contract_rejects_unsupported_semantics(override, match):
    with pytest.raises(ValueError, match=match):
        moe_w2_cubit._layer_contract(_contract_layer(**override))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs pinned arena")
def test_tiered_read_failure_never_publishes_stale_hit(pack_env, monkeypatch):
    seed = _built_store(pack_env)
    seed.release()
    store = moe_w2_store.TieredPackStore(
        str(pack_env), "base", n_layers=2, n_experts=2,
        slot_bytes=16, ram_gb=0.001
    )
    original = store._read_row

    def fail_after_write(slot, off):
        store._arena[slot, :16].fill_(0xA5)
        raise IOError("injected read failure")

    monkeypatch.setattr(store, "_read_row", fail_after_write)
    with pytest.raises(IOError, match="injected"):
        store.rows_for([(0, 0)])
    assert (0, 0) not in store._pos
    assert all(owner != (0, 0) for owner in store._owner_pair)

    monkeypatch.setattr(store, "_read_row", original)
    row = store.rows_for([(0, 0)])[0]
    expected = torch.cat(_parts(), dim=1)[0]
    assert torch.equal(row.cpu(), expected)
    store.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tier")
def test_delta_tier_close_stops_manager_and_releases_store(monkeypatch):
    monkeypatch.delenv("VLLM_MOE_W2_STORE_DIR", raising=False)
    tier = moe_w2_delta.DeltaTier(
        2, 4, torch.device("cuda"), w13_bytes=2048, w2_bytes=2048,
        pool_gb=0.001, tag="close-test")
    tier.start()
    thread = tier._thread
    assert thread is not None and thread.is_alive()
    tier.close()
    assert tier._thread is None
    assert not thread.is_alive()
    assert len(tier._store) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tier")
def test_step_end_snapshots_seen_before_manager_clears_it(monkeypatch):
    monkeypatch.delenv("VLLM_MOE_W2_STORE_DIR", raising=False)
    tier = moe_w2_delta.DeltaTier(
        2, 4, torch.device("cuda"), w13_bytes=2048, w2_bytes=2048,
        pool_gb=0.001, tag="snapshot-test")
    tier.start()
    tier.seen[0, 1] = 3
    tier.step_end()
    tier.wait_manager_idle()
    assert int(tier.seen.sum()) == 0
    assert float(tier._freq[0, 1]) > 0
    assert tier._win_active > 0
    tier.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_unpermute_is_deterministic_and_close_to_legacy():
    tokens, top_k, hidden = 16, 6, 510
    routes = tokens * top_k
    slots = routes + 13
    sorted_ids = torch.cat([
        torch.randperm(routes, device="cuda", dtype=torch.int64),
        torch.full((13,), routes + 1, device="cuda", dtype=torch.int64),
    ])
    c2 = torch.randn(
        slots, hidden, device="cuda", dtype=torch.bfloat16)
    weights_storage = torch.rand(tokens, top_k * 2, device="cuda")
    weights = weights_storage[:, ::2]  # exercise non-contiguous stride
    row_mask = torch.randint(
        0, 2, (slots,), device="cuda", dtype=torch.uint8)
    valid = sorted_ids < routes
    sorted_weights = weights.reshape(-1)[
        sorted_ids.clamp(max=routes - 1)]
    sorted_weights = torch.where(
        valid, sorted_weights, torch.zeros_like(sorted_weights)).float()
    sorted_weights *= row_mask.float()
    dst = torch.where(
        valid, sorted_ids, torch.full_like(sorted_ids, routes)).long()
    gathered = torch.zeros(
        routes + 1, hidden, device="cuda", dtype=torch.float32)
    gathered.index_copy_(
        0, dst, c2.float() * sorted_weights.unsqueeze(1))
    legacy = gathered[:routes].view(
        tokens, top_k, hidden).sum(1).to(torch.bfloat16)

    inverse = torch.empty(routes, device="cuda", dtype=torch.int32)
    moe_w2_cubit._invert_sorted_ids_kernel[
        (triton.cdiv(slots, 256),)
    ](sorted_ids, inverse, slots, routes, BLOCK=256)

    def run():
        out = torch.empty(
            tokens, hidden, device="cuda", dtype=torch.bfloat16)
        moe_w2_cubit._deterministic_unpermute_kernel[
            (tokens, triton.cdiv(hidden, 256))
        ](
            c2, weights, inverse, row_mask, out, hidden, c2.stride(0),
            weights.stride(0), weights.stride(1), out.stride(0),
            TOP_K=top_k, HAS_ROW_MASK=True, BLOCK_H=256, num_warps=4)
        return out

    fused_a, fused_b = run(), run()
    torch.cuda.synchronize()
    assert torch.equal(fused_a, fused_b)
    torch.testing.assert_close(
        fused_a.float(), legacy.float(), rtol=2e-3, atol=2e-2)

    graph_out = torch.empty(
        tokens, hidden, device="cuda", dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        moe_w2_cubit._invert_sorted_ids_kernel[
            (triton.cdiv(slots, 256),)
        ](sorted_ids, inverse, slots, routes, BLOCK=256)
        moe_w2_cubit._deterministic_unpermute_kernel[
            (tokens, triton.cdiv(hidden, 256))
        ](
            c2, weights, inverse, row_mask, graph_out, hidden,
            c2.stride(0), weights.stride(0), weights.stride(1),
            graph_out.stride(0), TOP_K=top_k, HAS_ROW_MASK=True,
            BLOCK_H=256, num_warps=4)
    graph.replay()
    replay_a = graph_out.clone()
    graph.replay()
    replay_b = graph_out.clone()
    torch.cuda.synchronize()
    assert torch.equal(replay_a, replay_b)

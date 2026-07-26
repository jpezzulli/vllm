import torch

from vllm.v1.worker.gpu.spec_decode.dspark.scheduler import allocate_widths


def test_dspark_threshold_width_never_exceeds_scheduled_length():
    survival = torch.ones((2, 4), dtype=torch.float32)
    calibration = torch.ones(4, dtype=torch.float32)
    widths = allocate_widths(
        survival, calibration, num_reqs=2, length=2,
        tau=0.5, budget_frac=1.0)
    assert torch.equal(widths, torch.tensor([2, 2], dtype=torch.int32))


def test_dspark_tied_budget_never_inflates():
    survival = torch.ones((3, 4), dtype=torch.float32)
    calibration = torch.ones(4, dtype=torch.float32)
    widths = allocate_widths(
        survival, calibration, num_reqs=3, length=4,
        tau=0.0, budget_frac=0.25)
    assert int(widths.sum()) == 3
    assert bool((widths <= 4).all())


def test_dspark_zero_budget_allocates_no_verify_tail():
    survival = torch.rand((2, 4), dtype=torch.float32)
    calibration = torch.ones(4, dtype=torch.float32)
    widths = allocate_widths(
        survival, calibration, num_reqs=2, length=4,
        tau=0.0, budget_frac=0.0)
    assert torch.equal(widths, torch.zeros(2, dtype=torch.int32))


def test_dspark_calibration_is_projected_to_valid_prefix():
    survival = torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float32)
    calibration = torch.tensor([1.0, 0.1, 2.0, 2.0], dtype=torch.float32)
    widths = allocate_widths(
        survival, calibration, num_reqs=1, length=4,
        tau=0.15, budget_frac=1.0)
    # Calibrated scores are [0.9, .08, 1.4, 1.2]; cummin projects them to
    # [.9, .08, .08, .08], so the threshold cannot select a disjoint tail.
    assert torch.equal(widths, torch.tensor([1], dtype=torch.int32))

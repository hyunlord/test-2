import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sparsetap.gd_models import (
    GDResult,
    run_cos_parity_gd,
    run_gumbel_mask_gd,
    run_next_bit_cnn,
    run_next_bit_mlp,
    run_reinforce,
    run_soft_mask_gd,
)


def _make_synthetic(taps, N=50, L=64, noise_p=0.0, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows = []
    max_tap = max(taps)
    for _ in range(N):
        seq = np.zeros(L, dtype=np.uint8)
        seq[:max_tap] = rng.integers(0, 2, size=max_tap, dtype=np.uint8)
        for n in range(max_tap, L):
            bit = 0
            for tap in taps:
                bit ^= int(seq[n - tap])
            if noise_p > 0.0 and rng.random() < noise_p:
                bit ^= 1
            seq[n] = bit
        rows.append(seq)
    data_01 = np.asarray(rows, dtype=np.uint8)
    data_pm = (1.0 - 2.0 * data_01).astype(np.float32)
    return data_01, data_pm


def test_gd_result_dataclass():
    result = GDResult()
    assert result.method == ""
    assert result.offsets == []
    assert result.loss_history == []


def test_soft_mask_gd_runs():
    _, data_pm = _make_synthetic([1, 3], N=50, L=64)
    result = run_soft_mask_gd(data_pm, max_w=8, n_epochs=5, batch_size=16, seed=42)
    assert isinstance(result, GDResult)
    assert isinstance(result.offsets, list)
    assert len(result.loss_history) > 0


def test_cos_parity_gd_runs():
    data_01, data_pm = _make_synthetic([1, 3], N=50, L=64)
    result = run_cos_parity_gd(data_01, data_pm, max_w=8, n_steps=50, batch_size=64, seed=42)
    assert isinstance(result, GDResult)
    assert len(result.loss_history) > 0


def test_gumbel_mask_gd_runs():
    _, data_pm = _make_synthetic([1, 3], N=50, L=64)
    result = run_gumbel_mask_gd(data_pm, max_w=8, n_steps=50, batch_size=16, seed=42)
    assert isinstance(result, GDResult)
    assert len(result.loss_history) > 0


def test_reinforce_runs():
    _, data_pm = _make_synthetic([1, 3], N=50, L=64)
    result = run_reinforce(data_pm, max_w=8, n_steps=50, batch_size=16, n_samples=8, seed=42)
    assert isinstance(result, GDResult)
    assert len(result.loss_history) > 0


def test_next_bit_mlp_runs():
    data_01, _ = _make_synthetic([1, 3], N=50, L=64)
    result = run_next_bit_mlp(data_01, max_w=8, n_epochs=3, batch_size=64, seed=42)
    assert isinstance(result, GDResult)
    assert 0.0 <= result.accuracy <= 1.0


def test_next_bit_cnn_runs():
    data_01, _ = _make_synthetic([1, 3], N=50, L=64)
    result = run_next_bit_cnn(data_01, max_w=8, n_epochs=3, batch_size=64, seed=42)
    assert isinstance(result, GDResult)
    assert 0.0 <= result.accuracy <= 1.0


def test_soft_mask_gd_finds_simple_taps():
    _, data_pm = _make_synthetic([1, 3], N=200, L=32, noise_p=0.0)
    result = run_soft_mask_gd(data_pm, max_w=8, n_epochs=200, batch_size=128, lr=0.1, seed=42)
    assert result.bias > 0.3

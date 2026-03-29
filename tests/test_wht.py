import numpy as np

from sparsetap.wht import expected_noise_floor, run_wht_scan, vectorized_wht


def _make_synthetic(taps, N=100, L=64, noise_p=0.0, seed=42):
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


def test_vectorized_wht_finds_known_taps():
    data_01, data_pm = _make_synthetic([3, 7], N=100, L=64, noise_p=0.0)
    F = vectorized_wht(data_01, data_pm, w=8, verbose=False)
    mask = (1 << 2) | (1 << 6)
    assert abs(F[mask]) > 0.5
    assert np.argmax(np.abs(F)) == mask


def test_wht_noise_floor():
    rng = np.random.default_rng(42)
    data_01 = rng.integers(0, 2, size=(100, 64), dtype=np.uint8)
    data_pm = (1.0 - 2.0 * data_01).astype(np.float32)
    F = vectorized_wht(data_01, data_pm, w=8, verbose=False)
    sigma, expected_max = expected_noise_floor(w=8, N_seq=100, L_seq=64)
    assert np.max(np.abs(F)) < expected_max * 4.0
    assert sigma > 0


def test_expected_noise_floor_calculation():
    sigma, expected_max = expected_noise_floor(w=16, N_seq=100)
    assert np.isclose(sigma, 1.0 / np.sqrt(100 * (256 - 16)))
    assert expected_max > sigma


def test_run_wht_scan_returns_correct_structure():
    data_01, data_pm = _make_synthetic([2, 5], N=30, L=48, noise_p=0.0)
    result = run_wht_scan(data_01, data_pm, w_range=[8], prefix=data_01[0, :16], verbose=False)
    assert "scan_results" in result
    assert "best_offsets" in result
    assert "signal_found" in result
    assert result["scan_results"][0]["w"] == 8

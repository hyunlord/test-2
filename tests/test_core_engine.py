import numpy as np

from sparsetap_core import (
    _build_lagged_matrix,
    _mask_to_taps,
    _taps_to_mask,
    check_prefix_consistency,
    compute_log_likelihood,
    compute_log_likelihood_fast,
    compute_log_likelihood_slow_from_matrix,
    local_search,
    optimized_local_search,
)


def rollout(prefix, taps, total_len):
    seq = np.zeros(total_len, dtype=np.uint8)
    seq[: len(prefix)] = prefix
    for n in range(len(prefix), total_len):
        value = 0
        for tap in taps:
            value ^= int(seq[n - tap])
        seq[n] = value
    return seq


def make_dataset(taps, num_sequences=12, total_len=48, noise=0.0):
    max_tap = max(taps)
    rng = np.random.default_rng(123)
    rows = []
    for seq_idx in range(num_sequences):
        prefix = rng.integers(0, 2, size=max_tap, dtype=np.uint8)
        seq = rollout(prefix, taps, total_len)
        if noise:
            flip = ((np.arange(total_len) + seq_idx) % int(round(1 / noise)) == 0).astype(np.uint8)
            flip[:max_tap] = 0
            seq = np.bitwise_xor(seq, flip)
        rows.append(seq)
    return np.asarray(rows, dtype=np.uint8)


def test_check_prefix_consistency_returns_true_for_matching_rule():
    prefix = rollout(np.array([1, 0, 1], dtype=np.uint8), [1, 3], 16)
    assert check_prefix_consistency([1, 3], prefix)


def test_check_prefix_consistency_returns_false_for_wrong_rule():
    prefix = rollout(np.array([1, 0, 1], dtype=np.uint8), [1, 3], 16)
    assert not check_prefix_consistency([2, 4], prefix)


def test_compute_log_likelihood_prefers_true_taps():
    data = make_dataset([1, 3], noise=0.0)
    true_score = compute_log_likelihood([1, 3], data, noise=0.2)
    wrong_score = compute_log_likelihood([2, 4], data, noise=0.2)
    assert true_score > wrong_score


def test_fast_matrix_log_likelihood_matches_slow_reference():
    taps = [1, 3]
    mask = _taps_to_mask(taps)
    data = make_dataset(taps, num_sequences=10, total_len=24, noise=0.0)
    X, y = _build_lagged_matrix(data, max_lag=8)

    fast_score = compute_log_likelihood_fast(mask, X, y, noise=0.2)
    slow_score = compute_log_likelihood_slow_from_matrix(mask, X, y, noise=0.2)

    assert np.isclose(fast_score, slow_score)


def test_local_search_recovers_simple_rule_from_clean_data():
    taps = [1, 2, 5]
    prefix = rollout(np.array([1, 0, 1, 1, 0], dtype=np.uint8), taps, 64)
    data = make_dataset(taps, num_sequences=32, total_len=80, noise=0.0)
    true_score = compute_log_likelihood(taps, data, noise=0.2)

    best_taps, best_score = local_search(
        data,
        prefix,
        max_lag=8,
        max_taps=5,
        restarts=6,
        steps=400,
        seed=42,
    )

    assert check_prefix_consistency(best_taps, prefix)
    assert best_score >= true_score
    assert np.isfinite(best_score)


def test_optimized_local_search_returns_mask_and_valid_score():
    taps = [1, 2, 5]
    prefix = rollout(np.array([1, 0, 1, 1, 0], dtype=np.uint8), taps, 64)
    data = make_dataset(taps, num_sequences=32, total_len=80, noise=0.0)

    best_mask, best_score = optimized_local_search(
        data,
        prefix,
        max_lag=8,
        max_taps=5,
        restarts=4,
        steps=300,
        seed=42,
    )

    best_taps = _mask_to_taps(best_mask, max_lag=8)
    assert isinstance(best_mask, int)
    assert best_mask == _taps_to_mask(best_taps)
    assert check_prefix_consistency(best_taps, prefix)
    assert np.isfinite(best_score)

import json
from pathlib import Path

import numpy as np

from run_final_search import extract_answer, mask_to_taps, rollout, run_search_experiments
from sparsetap_core import _taps_to_mask


def make_sequence(prefix, taps, total_len=64):
    seq = np.zeros(total_len, dtype=np.uint8)
    seq[: len(prefix)] = prefix
    for idx in range(len(prefix), total_len):
        value = 0
        for tap in taps:
            value ^= int(seq[idx - tap])
        seq[idx] = value
    return seq


def make_dataset(taps, num_sequences=12, total_len=64):
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(num_sequences):
        prefix = rng.integers(0, 2, size=max(taps), dtype=np.uint8)
        rows.append(make_sequence(prefix, taps, total_len=total_len))
    return np.asarray(rows, dtype=np.uint8)


def test_mask_to_taps_roundtrip():
    taps = [1, 3, 7]
    mask = _taps_to_mask(taps)
    assert mask_to_taps(mask) == taps


def test_rollout_and_extract_answer():
    taps = [1, 3]
    prefix = np.array([1, 0, 1], dtype=np.uint8)
    seq = rollout(prefix, taps, total_len=10)
    assert seq.shape == (10,)
    assert seq[:3].tolist() == prefix.tolist()
    assert extract_answer(np.arange(256) % 2) == "".join(str(i % 2) for i in range(64, 256))


def test_run_search_experiments_returns_ranked_candidates():
    taps = [1, 2, 5]
    prefix = make_sequence(np.array([1, 0, 1, 1, 0], dtype=np.uint8), taps, total_len=64)
    data = make_dataset(taps, num_sequences=16, total_len=80)

    candidates = run_search_experiments(
        data,
        prefix,
        num_runs=3,
        max_lag=8,
        max_taps=5,
        restarts=2,
        steps=100,
        base_seed=42,
    )

    assert len(candidates) == 3
    assert candidates[0]["score"] >= candidates[-1]["score"]
    assert all("mask" in candidate for candidate in candidates)
    assert all("taps" in candidate for candidate in candidates)
    assert all(candidate["score"] > float("-inf") for candidate in candidates)

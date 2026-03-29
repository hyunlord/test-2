import json
from pathlib import Path

import numpy as np

from sparsetap.data import build_dataset, split_dataset_by_sequence, temporal_train_validation_split
from sparsetap.evaluation import evaluate_candidate, rank_candidates_table
from sparsetap.utils import extract_answer_bits, log_failed_attempt, mask_to_taps, rollout_sequence, taps_to_mask


def make_clean_data(taps, num_sequences=12, length=80):
    rng = np.random.default_rng(123)
    rows = []
    for _ in range(num_sequences):
        seq = np.zeros(length, dtype=np.uint8)
        seq[: max(taps)] = rng.integers(0, 2, size=max(taps), dtype=np.uint8)
        for pos in range(max(taps), length):
            value = 0
            for tap in taps:
                value ^= int(seq[pos - tap])
            seq[pos] = value
        rows.append(seq)
    return np.asarray(rows, dtype=np.uint8)


def test_mask_roundtrip_and_rollout():
    taps = [1, 4, 7]
    mask = taps_to_mask(taps)
    prefix = np.array([1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
    seq = rollout_sequence(prefix, taps, total_length=20)

    assert mask_to_taps(mask) == taps
    assert seq.shape == (20,)
    assert extract_answer_bits(np.arange(256) % 2, start=64, end=68) == "0101"


def test_sequence_and_temporal_splits():
    data = make_clean_data([1, 3], num_sequences=10, length=80)
    train, holdout = split_dataset_by_sequence(data, holdout_fraction=0.2, seed=42)
    dataset = build_dataset(data, max_lag=8)
    temporal = temporal_train_validation_split(dataset, validation_fraction=0.25)

    assert train.shape[0] == 8
    assert holdout.shape[0] == 2
    assert temporal["X_train"].shape[0] + temporal["X_valid"].shape[0] == dataset["X"].shape[0]


def test_evaluation_and_failed_attempt_logging(tmp_path):
    data = make_clean_data([1, 3], num_sequences=8, length=40)
    prefix = data[0, :16]
    candidate = {"taps": [1, 3], "method": "unit", "track": "rule_recovery", "metadata": {}}
    scored = evaluate_candidate(candidate, data=data, prefix=prefix, noise=0.2)
    table = rank_candidates_table([scored])
    log_path = tmp_path / "failed_attempts.jsonl"
    log_failed_attempt(
        log_path,
        method="plain_linear",
        config={"C": 0.1},
        result={"score": -123.4},
        note="Weak under exact prefix filtering.",
        promising=False,
    )

    assert scored["scores"]["prefix_consistency"] == 1
    assert "rank_tuple" in scored["scores"]
    assert table.shape[0] == 1
    payload = json.loads(log_path.read_text().splitlines()[0])
    assert payload["method"] == "plain_linear"
    assert payload["promising"] is False

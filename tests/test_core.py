import numpy as np

from sparsetap.data import build_dataset
from sparsetap.scoring import (
    check_prefix_consistency,
    compute_accuracy,
    compute_log_likelihood,
    rank_candidates,
)
from sparsetap.search import rollout_bits
from sparsetap_utils import (
    build_supervised_dataset,
    evaluate_rule_candidate,
    prefix_to_array,
)


def make_clean_sequences(taps, num_sequences=8, length=24, warmup=None):
    if warmup is None:
        warmup = max(taps)
    sequences = []
    for seed_idx in range(num_sequences):
        base = np.array([(seed_idx + bit) % 2 for bit in range(length)], dtype=np.uint8)
        for n in range(warmup, length):
            base[n] = 0
            for tap in taps:
                base[n] ^= base[n - tap]
        sequences.append(base)
    return np.array(sequences, dtype=np.uint8)


def test_build_dataset_shapes_and_values():
    data = make_clean_sequences([1, 3], num_sequences=2, length=10, warmup=4)
    dataset = build_dataset(data, max_lag=4)

    assert dataset["X"].shape == (12, 4)
    assert dataset["y"].shape == (12,)
    assert dataset["X"][0].tolist() == data[0, 0:4].tolist()[::-1]
    assert dataset["y"][0] == data[0, 4]


def test_prefix_consistency_accepts_true_rule_and_rejects_wrong_rule():
    prefix = rollout_bits([1, 3], np.array([1, 0, 1], dtype=np.uint8), total_length=8)
    consistent, generated = check_prefix_consistency([1, 3], prefix, return_generated=True)
    inconsistent, _ = check_prefix_consistency([2, 4], prefix, return_generated=True)

    assert consistent == 1
    assert inconsistent == 0
    assert len(generated) == len(prefix)


def test_likelihood_and_accuracy_reward_true_taps():
    data = make_clean_sequences([1, 3], num_sequences=10, length=20)
    ll_true = compute_log_likelihood([1, 3], data, noise=0.2)
    ll_wrong = compute_log_likelihood([2, 4], data, noise=0.2)
    acc_true = compute_accuracy([1, 3], data)
    acc_wrong = compute_accuracy([2, 4], data)

    assert ll_true > ll_wrong
    assert acc_true > acc_wrong
    assert acc_true == 1.0


def test_rollout_bits_generates_requested_length():
    prefix = np.array([1, 0, 1, 1], dtype=np.uint8)
    generated = rollout_bits([1, 3], prefix, total_length=10)

    assert generated.shape == (10,)
    assert generated[:4].tolist() == prefix.tolist()


def test_rank_candidates_sorts_by_shared_tuple():
    candidates = [
        {
            "taps": [1, 3],
            "method": "a",
            "scores": {
                "prefix_consistency": 1,
                "log_likelihood": -5.0,
                "accuracy": 0.8,
            },
        },
        {
            "taps": [1],
            "method": "b",
            "scores": {
                "prefix_consistency": 1,
                "log_likelihood": -5.0,
                "accuracy": 0.8,
            },
        },
    ]

    ranked = rank_candidates(candidates)

    assert ranked[0]["method"] == "b"
    assert ranked[0]["scores"]["rank_tuple"] > ranked[1]["scores"]["rank_tuple"]


def test_top_level_utils_match_required_api():
    arr = prefix_to_array("0" * 64)
    assert arr.shape == (64,)

    data = make_clean_sequences([1, 3], num_sequences=2, length=70)
    dataset = build_supervised_dataset(data, max_lag=64)
    assert dataset["X"].shape[1] == 64

    candidate = {
        "candidate_id": "test",
        "track": "rule_recovery",
        "method": "manual",
        "taps": [1, 3],
        "W": 3,
        "scores": {},
        "metadata": {},
    }
    scored = evaluate_rule_candidate(candidate, data, rollout_bits([1, 3], np.array([1, 0, 1], dtype=np.uint8), total_length=64))
    assert scored["scores"]["num_taps"] == 2

import json
import random
from pathlib import Path

import numpy as np

from sparsetap_config import DEFAULT_CONFIG
from sparsetap_core import _mask_to_taps, check_prefix_consistency, optimized_local_search
from sparsetap_utils import load_sequences, prefix_to_array


def mask_to_taps(mask):
    return _mask_to_taps(int(mask), max_lag=64)


def rollout(prefix, taps, total_len=256):
    prefix = np.asarray(prefix, dtype=np.uint8)
    taps = sorted(int(tap) for tap in taps)
    seq = np.zeros(total_len, dtype=np.uint8)
    seq[: len(prefix)] = prefix
    start = max(taps)

    for idx in range(len(prefix), total_len):
        if idx < start:
            continue
        value = 0
        for tap in taps:
            value ^= int(seq[idx - tap])
        seq[idx] = value
    return seq


def extract_answer(seq):
    seq = np.asarray(seq, dtype=np.uint8)
    return "".join(str(int(bit)) for bit in seq[64:256])


def _candidate_from_result(mask, score, seed):
    taps = mask_to_taps(mask)
    return {
        "seed": int(seed),
        "mask": int(mask),
        "taps": taps,
        "score": float(score),
        "num_taps": len(taps),
    }


def run_search_experiments(
    data,
    prefix,
    num_runs=20,
    max_lag=64,
    max_taps=16,
    restarts=20,
    steps=2000,
    base_seed=42,
):
    random.seed(base_seed)
    np.random.seed(base_seed)

    data = np.asarray(data, dtype=np.uint8)
    prefix = np.asarray(prefix, dtype=np.uint8)

    candidates = []
    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        mask, score = optimized_local_search(
            data,
            prefix,
            max_lag=max_lag,
            max_taps=max_taps,
            restarts=restarts,
            steps=steps,
            seed=seed,
            noise=0.2,
        )
        taps = mask_to_taps(mask)
        print(f"run_seed={seed} best_score={score} taps={taps}")

        if not taps or not check_prefix_consistency(taps, prefix):
            continue

        candidate = _candidate_from_result(mask, score, seed)
        candidates.append(candidate)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def save_outputs(candidates, answer, answer_path="answer.txt", candidates_path="candidates.json"):
    Path(answer_path).write_text(answer + "\n")
    Path(candidates_path).write_text(json.dumps(candidates, indent=2))


def main():
    config = DEFAULT_CONFIG
    random.seed(config.seed)
    np.random.seed(config.seed)

    data = load_sequences(str(config.data_path))
    prefix = prefix_to_array(config.test_prefix)

    candidates = run_search_experiments(
        data,
        prefix,
        num_runs=20,
        max_lag=config.max_lag,
        max_taps=config.max_taps,
        restarts=20,
        steps=2000,
        base_seed=config.seed,
    )
    if not candidates:
        raise RuntimeError("No valid candidates found.")

    print("\nTop 5 candidates")
    for idx, candidate in enumerate(candidates[:5], start=1):
        print(
            f"{idx}. taps={candidate['taps']} "
            f"score={candidate['score']} "
            f"num_taps={candidate['num_taps']}"
        )

    best = candidates[0]
    best_taps = best["taps"]
    best_score = best["score"]
    seq = rollout(prefix, best_taps, total_len=256)
    answer = extract_answer(seq)

    print("\nBest candidate")
    print(f"best_taps={best_taps}")
    print(f"best_score={best_score}")
    print(f"num_taps={len(best_taps)}")
    print("final_answer=")
    print(answer)

    save_outputs(candidates, answer, answer_path="answer.txt", candidates_path="candidates.json")


if __name__ == "__main__":
    main()

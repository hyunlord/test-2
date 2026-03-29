from itertools import combinations

import numpy as np

from .scoring import compute_accuracy, score_candidate


def _single_lag_stat(data, lag):
    target = data[:, lag:]
    source = data[:, :-lag]
    match_rate = float((target == source).mean())
    lift = abs(match_rate - 0.5)
    return match_rate, lift


def run_single_lag_scan(data, prefix, max_lag=64, top_k=12, noise=0.2):
    records = []
    candidates = []
    for lag in range(1, max_lag + 1):
        match_rate, lift = _single_lag_stat(data, lag)
        candidate = {
            "taps": [lag],
            "method": "single_lag_scan",
            "scores": {
                "single_lag_match_rate": match_rate,
                "single_lag_lift": lift,
            },
            "metadata": {"lag": lag},
        }
        candidate = score_candidate(candidate, data=data, prefix=prefix, noise=noise)
        records.append(candidate)
    records.sort(key=lambda item: item["scores"]["single_lag_lift"], reverse=True)
    candidates.extend(records[:top_k])
    return {"records": records, "candidates": candidates}


def run_pair_scan(data, prefix, max_lag=64, top_k=24, top_lags=None, noise=0.2):
    if top_lags is None:
        single = run_single_lag_scan(data, prefix, max_lag=max_lag, top_k=min(16, max_lag), noise=noise)
        top_lags = [cand["taps"][0] for cand in single["candidates"]]
    else:
        top_lags = list(top_lags)

    records = []
    for lag_a, lag_b in combinations(sorted(set(top_lags)), 2):
        start = max(lag_a, lag_b)
        target = data[:, start:]
        predictor = np.bitwise_xor(data[:, start - lag_a : -lag_a or None], data[:, start - lag_b : -lag_b or None])
        match_rate = float((target == predictor).mean())
        candidate = {
            "taps": [lag_a, lag_b],
            "method": "pair_scan",
            "scores": {
                "pair_match_rate": match_rate,
                "pair_lift": abs(match_rate - 0.5),
            },
            "metadata": {"lags": [lag_a, lag_b]},
        }
        candidate = score_candidate(candidate, data=data, prefix=prefix, noise=noise)
        records.append(candidate)

    records.sort(key=lambda item: item["scores"]["pair_lift"], reverse=True)
    return {"records": records, "candidates": records[:top_k]}


def summarize_tap_support(candidates, max_lag=64):
    support = np.zeros(max_lag + 1, dtype=np.float32)
    for rank, candidate in enumerate(candidates, start=1):
        weight = 1.0 / rank
        for tap in candidate.get("taps", []):
            if 1 <= tap <= max_lag:
                support[tap] += weight
    return support

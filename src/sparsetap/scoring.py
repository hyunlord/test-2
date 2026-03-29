import math

import numpy as np


def _predict_from_taps(sequence, taps, start):
    predictions = np.zeros(len(sequence) - start, dtype=np.uint8)
    for idx, pos in enumerate(range(start, len(sequence))):
        value = 0
        for tap in taps:
            value ^= int(sequence[pos - tap])
        predictions[idx] = value
    return predictions


def check_prefix_consistency(taps, prefix, return_generated=False):
    prefix = np.asarray(prefix, dtype=np.uint8)
    if len(prefix) == 0:
        return (1, prefix.copy()) if return_generated else 1
    if not taps:
        generated = np.zeros_like(prefix)
        generated[: min(1, len(prefix))] = prefix[: min(1, len(prefix))]
        consistent = int(np.array_equal(generated, prefix))
        return (consistent, generated) if return_generated else consistent

    max_tap = max(taps)
    if len(prefix) <= max_tap:
        generated = prefix.copy()
        return (1, generated) if return_generated else 1

    generated = prefix.copy()
    for pos in range(max_tap, len(prefix)):
        value = 0
        for tap in taps:
            value ^= int(generated[pos - tap])
        generated[pos] = value
    consistent = int(np.array_equal(generated, prefix))
    return (consistent, generated) if return_generated else consistent


def compute_log_likelihood(taps, data, noise=0.2):
    data = np.asarray(data, dtype=np.uint8)
    if not taps:
        return float("-inf")
    max_tap = max(taps)
    if data.shape[1] <= max_tap:
        return float("-inf")

    log_p_match = math.log(max(1e-12, 1.0 - noise))
    log_p_flip = math.log(max(1e-12, noise))
    total = 0.0
    for sequence in data:
        predicted = _predict_from_taps(sequence, taps, max_tap)
        observed = sequence[max_tap:]
        matches = predicted == observed
        total += matches.sum() * log_p_match + (~matches).sum() * log_p_flip
    return float(total)


def compute_accuracy(taps, data):
    data = np.asarray(data, dtype=np.uint8)
    if not taps:
        return 0.0
    max_tap = max(taps)
    if data.shape[1] <= max_tap:
        return 0.0
    matches = 0
    total = 0
    for sequence in data:
        predicted = _predict_from_taps(sequence, taps, max_tap)
        observed = sequence[max_tap:]
        matches += int((predicted == observed).sum())
        total += len(observed)
    return float(matches / total) if total else 0.0


def score_candidate(candidate, data, prefix, noise=0.2):
    taps = sorted(set(int(tap) for tap in candidate.get("taps", [])))
    candidate["taps"] = taps
    scores = dict(candidate.get("scores", {}))
    scores["prefix_consistency"] = check_prefix_consistency(taps, prefix)
    scores["log_likelihood"] = compute_log_likelihood(taps, data, noise=noise)
    scores["accuracy"] = compute_accuracy(taps, data)
    scores["num_taps"] = len(taps)
    scores["rank_tuple"] = (
        scores["prefix_consistency"],
        scores["log_likelihood"],
        scores["accuracy"],
        -scores["num_taps"],
    )
    candidate["scores"] = scores
    return candidate


def rank_candidates(candidates, data=None, prefix=None, noise=0.2):
    ranked = []
    for candidate in candidates:
        if data is not None and prefix is not None:
            ranked.append(score_candidate(candidate, data=data, prefix=prefix, noise=noise))
        else:
            scores = dict(candidate.get("scores", {}))
            num_taps = scores.get("num_taps", len(candidate.get("taps", [])))
            scores["num_taps"] = num_taps
            scores["rank_tuple"] = (
                scores.get("prefix_consistency", 0),
                scores.get("log_likelihood", float("-inf")),
                scores.get("accuracy", 0.0),
                -num_taps,
            )
            candidate["scores"] = scores
            ranked.append(candidate)
    ranked.sort(key=lambda item: item["scores"]["rank_tuple"], reverse=True)
    return ranked

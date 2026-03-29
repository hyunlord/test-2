import pandas as pd

from .scoring import check_prefix_consistency, compute_accuracy, compute_log_likelihood


def evaluate_candidate(candidate, data, prefix, noise=0.2):
    candidate = dict(candidate)
    taps = sorted(set(int(t) for t in candidate.get("taps", [])))
    scores = dict(candidate.get("scores", {}))
    scores["prefix_consistency"] = int(check_prefix_consistency(taps, prefix))
    scores["log_likelihood"] = float(compute_log_likelihood(taps, data, noise=noise)) if taps else float("-inf")
    scores["accuracy"] = float(compute_accuracy(taps, data)) if taps else 0.0
    scores["num_taps"] = len(taps)
    scores["rank_tuple"] = (
        scores["prefix_consistency"],
        scores["log_likelihood"],
        scores["accuracy"],
        -scores["num_taps"],
    )
    candidate["taps"] = taps
    candidate["scores"] = scores
    candidate["W"] = max(taps) if taps else None
    return candidate


def rank_candidates(candidates, data=None, prefix=None, noise=0.2):
    ranked = []
    for candidate in candidates:
        if data is not None and prefix is not None:
            ranked.append(evaluate_candidate(candidate, data=data, prefix=prefix, noise=noise))
        else:
            candidate = dict(candidate)
            scores = dict(candidate.get("scores", {}))
            scores["rank_tuple"] = (
                scores.get("prefix_consistency", 0),
                scores.get("log_likelihood", float("-inf")),
                scores.get("accuracy", 0.0),
                -scores.get("num_taps", len(candidate.get("taps", []))),
            )
            candidate["scores"] = scores
            ranked.append(candidate)
    ranked.sort(key=lambda item: item["scores"]["rank_tuple"], reverse=True)
    return ranked


def rank_candidates_table(candidates):
    ranked = rank_candidates(candidates)
    rows = []
    for idx, candidate in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": idx,
                "track": candidate.get("track"),
                "method": candidate.get("method"),
                "taps": candidate.get("taps"),
                "W": candidate.get("W"),
                "prefix_consistency": candidate.get("scores", {}).get("prefix_consistency"),
                "log_likelihood": candidate.get("scores", {}).get("log_likelihood"),
                "accuracy": candidate.get("scores", {}).get("accuracy"),
                "num_taps": candidate.get("scores", {}).get("num_taps"),
                "note": candidate.get("metadata", {}).get("note", ""),
            }
        )
    return pd.DataFrame(rows)

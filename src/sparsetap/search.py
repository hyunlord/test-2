from itertools import combinations

from .scoring import rank_candidates, score_candidate


def rollout_bits(taps, prefix, total_length=256):
    import numpy as np

    prefix = np.asarray(prefix, dtype=np.uint8)
    generated = np.zeros(total_length, dtype=np.uint8)
    generated[: len(prefix)] = prefix
    for pos in range(len(prefix), total_length):
        value = 0
        for tap in taps:
            value ^= int(generated[pos - tap])
        generated[pos] = value
    return generated


def run_prefix_solver(prefix, candidate_lags=None, max_taps=16):
    if candidate_lags is None:
        candidate_lags = range(1, min(64, len(prefix)) + 1)
    candidate_lags = [lag for lag in sorted(set(candidate_lags)) if lag < len(prefix)]

    candidates = []
    for size in range(1, min(max_taps, len(candidate_lags)) + 1):
        for taps in combinations(candidate_lags, size):
            consistent = score = None
            from .scoring import check_prefix_consistency

            consistent = check_prefix_consistency(list(taps), prefix)
            if consistent:
                candidates.append(
                    {
                        "taps": list(taps),
                        "method": "prefix_solver",
                        "scores": {"prefix_consistency": consistent},
                        "metadata": {"size": size},
                    }
                )
    return candidates


def _neighbors(taps, candidate_pool, max_taps):
    taps = tuple(sorted(set(taps)))
    pool = sorted(set(candidate_pool))
    seen = set()

    for lag in pool:
        if lag not in taps and len(taps) < max_taps:
            proposal = tuple(sorted(taps + (lag,)))
            if proposal not in seen:
                seen.add(proposal)
                yield proposal

    for lag in taps:
        proposal = tuple(item for item in taps if item != lag)
        if proposal and proposal not in seen:
            seen.add(proposal)
            yield proposal

    for old in taps:
        for new in pool:
            if new in taps:
                continue
            proposal = tuple(sorted((set(taps) - {old}) | {new}))
            if proposal not in seen:
                seen.add(proposal)
                yield proposal


def run_local_search(seed_taps, candidate_pool, data, prefix, noise=0.2, max_taps=16, max_rounds=30):
    current = {
        "taps": sorted(set(seed_taps)),
        "method": "local_search",
        "scores": {},
        "metadata": {"seed_taps": sorted(set(seed_taps))},
    }
    current = score_candidate(current, data=data, prefix=prefix, noise=noise)

    history = [current]
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        best_neighbor = current
        for taps in _neighbors(current["taps"], candidate_pool, max_taps=max_taps):
            candidate = {
                "taps": list(taps),
                "method": "local_search",
                "scores": {},
                "metadata": {"round": rounds},
            }
            candidate = score_candidate(candidate, data=data, prefix=prefix, noise=noise)
            history.append(candidate)
            if candidate["scores"]["rank_tuple"] > best_neighbor["scores"]["rank_tuple"]:
                best_neighbor = candidate
        if best_neighbor["scores"]["rank_tuple"] > current["scores"]["rank_tuple"]:
            current = best_neighbor
            improved = True
    return {"best": current, "history": rank_candidates(history)}


def run_beam_search(candidate_pool, data, prefix, noise=0.2, beam_width=12, max_taps=16):
    candidate_pool = sorted(set(candidate_pool))
    beam = [
        score_candidate(
            {
                "taps": [lag],
                "method": "beam_search",
                "scores": {},
                "metadata": {"depth": 1},
            },
            data=data,
            prefix=prefix,
            noise=noise,
        )
        for lag in candidate_pool
    ]
    beam = rank_candidates(beam)[:beam_width]
    all_candidates = list(beam)

    for depth in range(2, max_taps + 1):
        expanded = []
        for candidate in beam:
            used = set(candidate["taps"])
            for lag in candidate_pool:
                if lag in used:
                    continue
                proposal = sorted(candidate["taps"] + [lag])
                expanded.append(
                    score_candidate(
                        {
                            "taps": proposal,
                            "method": "beam_search",
                            "scores": {},
                            "metadata": {"depth": depth},
                        },
                        data=data,
                        prefix=prefix,
                        noise=noise,
                    )
                )
        if not expanded:
            break
        beam = rank_candidates(expanded)[:beam_width]
        all_candidates.extend(beam)
    return {"beam": beam, "candidates": rank_candidates(all_candidates)}


def run_greedy_search(candidate_pool, data, prefix, noise=0.2, max_taps=16):
    candidate_pool = sorted(set(candidate_pool))
    current_taps = []
    history = []

    best_candidate = score_candidate(
        {"taps": [], "method": "greedy_search", "scores": {}, "metadata": {"step": 0}},
        data=data,
        prefix=prefix,
        noise=noise,
    )
    history.append(best_candidate)

    for step in range(1, max_taps + 1):
        round_candidates = []
        for lag in candidate_pool:
            if lag in current_taps:
                continue
            proposal = sorted(current_taps + [lag])
            round_candidates.append(
                score_candidate(
                    {
                        "taps": proposal,
                        "method": "greedy_search",
                        "scores": {},
                        "metadata": {"step": step},
                    },
                    data=data,
                    prefix=prefix,
                    noise=noise,
                )
            )
        if not round_candidates:
            break
        best_round = rank_candidates(round_candidates)[0]
        if best_round["scores"]["rank_tuple"] > best_candidate["scores"]["rank_tuple"]:
            best_candidate = best_round
            current_taps = best_round["taps"]
            history.append(best_round)
        else:
            break

    return {"best": best_candidate, "history": rank_candidates(history)}


def run_reduced_exhaustive_search(candidate_pool, data, prefix, noise=0.2, max_combination_size=5):
    candidate_pool = sorted(set(candidate_pool))
    candidates = []
    for size in range(1, min(max_combination_size, len(candidate_pool)) + 1):
        for taps in combinations(candidate_pool, size):
            candidates.append(
                score_candidate(
                    {
                        "taps": list(taps),
                        "method": "reduced_exhaustive_search",
                        "scores": {},
                        "metadata": {"size": size},
                    },
                    data=data,
                    prefix=prefix,
                    noise=noise,
                )
            )
    return rank_candidates(candidates)

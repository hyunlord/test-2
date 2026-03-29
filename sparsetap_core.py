import math
import random

import numpy as np


def _normalize_taps(taps, max_lag=64):
    if isinstance(taps, (int, np.integer)):
        return _mask_to_taps(int(taps), max_lag=max_lag)
    normalized = sorted({int(tap) for tap in taps if 1 <= int(tap) <= max_lag})
    return normalized


def _taps_to_mask(taps):
    mask = 0
    for tap in taps:
        mask |= 1 << (tap - 1)
    return mask


def _mask_to_taps(mask, max_lag=64):
    return [lag for lag in range(1, max_lag + 1) if (mask >> (lag - 1)) & 1]


def _popcount(mask):
    return int(mask.bit_count())


def _bit_count_uint64(values):
    values = np.asarray(values, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    byte_view = values.view(np.uint8).reshape(values.shape + (8,))
    return np.unpackbits(byte_view, axis=-1).sum(axis=-1)


def check_prefix_consistency(taps, prefix):
    """
    taps: list of 1-based lags or 64-bit bitmask
    prefix: array of length 64

    Returns:
        True if taps exactly reproduce prefix recurrence
        False otherwise
    """
    prefix = np.asarray(prefix, dtype=np.uint8)
    taps = _normalize_taps(taps, max_lag=len(prefix))
    if not taps:
        return False

    start = max(taps)
    if start >= len(prefix):
        return True

    for n in range(start, len(prefix)):
        predicted = 0
        for tap in taps:
            predicted ^= int(prefix[n - tap])
        if predicted != int(prefix[n]):
            return False
    return True


def compute_log_likelihood(taps, data, noise=0.2):
    """
    taps: list of lags or 64-bit bitmask
    data: numpy array (N, 256)

    Returns:
        total log-likelihood (float)
    """
    data = np.asarray(data, dtype=np.uint8)
    taps = _normalize_taps(taps, max_lag=data.shape[1])
    if not taps:
        return float("-inf")

    start = max(taps)
    if start >= data.shape[1]:
        return float("-inf")

    predicted = np.zeros((data.shape[0], data.shape[1] - start), dtype=np.uint8)
    for tap in taps:
        predicted ^= data[:, start - tap : data.shape[1] - tap]

    observed = data[:, start:]
    matches = predicted == observed

    log_match = math.log1p(-noise)
    log_mismatch = math.log(noise)
    return float(matches.sum() * log_match + (~matches).sum() * log_mismatch)


def _build_lagged_matrix(data, max_lag=64):
    """
    Build the explicit lagged design matrix used by fast scoring.

    Returns:
        X: (samples, max_lag) with column j representing lag (j+1)
        y: (samples,)
    """
    data = np.asarray(data, dtype=np.uint8)
    if data.ndim != 2:
        raise ValueError("data must be a 2D binary array")
    seq_len = data.shape[1]
    if not (1 <= max_lag < seq_len):
        raise ValueError("max_lag must be between 1 and sequence_length - 1")

    rows = seq_len - max_lag
    X = np.empty((data.shape[0] * rows, max_lag), dtype=np.uint8)
    for lag in range(1, max_lag + 1):
        X[:, lag - 1] = data[:, max_lag - lag : seq_len - lag].reshape(-1)
    y = data[:, max_lag:].reshape(-1).astype(np.uint8)
    return X, y


def _build_lagged_bitmasks(data, max_lag=64):
    data = np.asarray(data, dtype=np.uint8)
    if data.ndim != 2:
        raise ValueError("data must be a 2D binary array")
    seq_len = data.shape[1]
    if not (1 <= max_lag < seq_len <= 256):
        raise ValueError("max_lag must be between 1 and sequence_length - 1")

    num_rows = data.shape[0] * (seq_len - max_lag)
    contexts = np.zeros(num_rows, dtype=np.uint64)
    y = data[:, max_lag:].reshape(-1).astype(np.uint8)

    for lag in range(1, max_lag + 1):
        bit = np.uint64(1 << (lag - 1))
        lag_values = data[:, max_lag - lag : seq_len - lag].reshape(-1).astype(np.uint64)
        contexts |= lag_values * bit

    return contexts, y


def _build_prefix_bitmasks(prefix, max_lag=64):
    prefix = np.asarray(prefix, dtype=np.uint8)
    usable_lag = min(max_lag, len(prefix))
    if usable_lag < 1:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.uint8)

    num_rows = len(prefix) - usable_lag
    if num_rows <= 0:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.uint8)

    contexts = np.zeros(num_rows, dtype=np.uint64)
    y = prefix[usable_lag:].astype(np.uint8)
    for lag in range(1, usable_lag + 1):
        bit = np.uint64(1 << (lag - 1))
        lag_values = prefix[usable_lag - lag : len(prefix) - lag].astype(np.uint64)
        contexts |= lag_values * bit
    return contexts, y


def _score_mask(mask, data, prefix, noise, max_lag):
    taps = _mask_to_taps(mask, max_lag=max_lag)
    if not taps:
        return float("-inf")
    if not check_prefix_consistency(taps, prefix):
        return float("-inf")
    return compute_log_likelihood(taps, data, noise=noise)


def _fast_prefix_consistency(mask, prefix_contexts, prefix_targets):
    if mask == 0:
        return False
    if prefix_targets.size == 0:
        return True
    parity = (_bit_count_uint64(prefix_contexts & np.uint64(mask)) & 1).astype(np.uint8)
    return bool(np.array_equal(parity, prefix_targets))


def _fast_log_likelihood(mask, contexts, y, noise=0.2):
    parity = (_bit_count_uint64(contexts & np.uint64(mask)) & 1).astype(np.uint8)
    matches = parity == y
    log_match = math.log1p(-noise)
    log_mismatch = math.log(noise)
    return float(matches.sum() * log_match + (~matches).sum() * log_mismatch)


def compute_log_likelihood_slow_from_matrix(mask, X, y, noise=0.2):
    """
    Slow, reference implementation over an explicit lagged matrix.
    Useful only for validation.
    """
    X = np.asarray(X, dtype=np.uint8)
    y = np.asarray(y, dtype=np.uint8)
    taps = _mask_to_taps(int(mask), max_lag=X.shape[1])
    if not taps:
        return float("-inf")

    selected = np.zeros(X.shape[0], dtype=np.uint8)
    for tap in taps:
        selected ^= X[:, tap - 1]
    matches = selected == y
    log_match = math.log1p(-noise)
    log_mismatch = math.log(noise)
    return float(matches.sum() * log_match + (~matches).sum() * log_mismatch)


def compute_log_likelihood_fast(mask, X, y, noise=0.2):
    """
    Fast log-likelihood from a precomputed lagged matrix.

    Parameters
    ----------
    mask : int
        64-bit mask whose bit i selects lag (i + 1).
    X : np.ndarray
        Binary lagged matrix of shape (samples, max_lag).
    y : np.ndarray
        Binary targets of shape (samples,).
    """
    mask = int(mask)
    X = np.asarray(X, dtype=np.uint8)
    y = np.asarray(y, dtype=np.uint8)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be a 1D array aligned with X")
    if mask == 0:
        return float("-inf")

    col_mask = np.array([(mask >> bit) & 1 for bit in range(X.shape[1])], dtype=np.uint8)
    parity = ((X @ col_mask) & 1).astype(np.uint8)
    matches = parity == y
    log_match = math.log1p(-noise)
    log_mismatch = math.log(noise)
    return float(matches.sum() * log_match + (~matches).sum() * log_mismatch)


def _score_mask_fast(mask, contexts, y, prefix_contexts, prefix_targets, noise):
    if mask == 0:
        return float("-inf")
    if not _fast_prefix_consistency(mask, prefix_contexts, prefix_targets):
        return float("-inf")
    return _fast_log_likelihood(mask, contexts, y, noise=noise)


def _score_key(mask, score):
    taps = _mask_to_taps(mask)
    return (score, -_popcount(mask), tuple(-tap for tap in taps))


def _random_mask(rng, max_lag, max_taps):
    size = int(rng.integers(1, max_taps + 1))
    taps = rng.choice(np.arange(1, max_lag + 1), size=size, replace=False)
    return _taps_to_mask(sorted(int(t) for t in taps))


def _heuristic_masks(data, prefix, max_lag, max_taps):
    candidates = []
    agreement = []
    for lag in range(1, max_lag + 1):
        score = float((data[:, lag:] == data[:, :-lag]).mean())
        agreement.append((abs(score - 0.5), lag))
    ranked = [lag for _, lag in sorted(agreement, reverse=True)]

    for size in range(1, min(max_taps, 4) + 1):
        taps = ranked[:size]
        mask = _taps_to_mask(taps)
        if check_prefix_consistency(taps, prefix):
            candidates.append(mask)

    for lag in ranked[: min(8, max_lag)]:
        mask = _taps_to_mask([lag])
        if check_prefix_consistency([lag], prefix):
            candidates.append(mask)
    return candidates


def _propose_neighbor(mask, rng, max_lag, max_taps):
    taps = set(_mask_to_taps(mask, max_lag=max_lag))
    available = [lag for lag in range(1, max_lag + 1) if lag not in taps]

    moves = []
    if len(taps) < max_taps and available:
        moves.append("add")
    if taps:
        moves.append("remove")
    if taps and available:
        moves.append("swap")
    if not moves:
        return mask

    move = moves[int(rng.integers(0, len(moves)))]
    new_taps = set(taps)

    if move == "add":
        new_taps.add(int(rng.choice(available)))
    elif move == "remove":
        new_taps.remove(int(rng.choice(list(taps))))
    else:
        new_taps.remove(int(rng.choice(list(taps))))
        new_taps.add(int(rng.choice(available)))

    return _taps_to_mask(sorted(new_taps))


def _propose_neighbor_mask(mask, rng, max_lag, max_taps):
    active = np.flatnonzero([(mask >> bit) & 1 for bit in range(max_lag)]).astype(np.int64)
    inactive = np.flatnonzero([((mask >> bit) & 1) == 0 for bit in range(max_lag)]).astype(np.int64)

    moves = []
    if active.size < max_taps and inactive.size > 0:
        moves.append(0)  # add
    if active.size > 0:
        moves.append(1)  # remove
    if active.size > 0 and inactive.size > 0:
        moves.append(2)  # swap
    if not moves:
        return mask

    move = moves[int(rng.integers(0, len(moves)))]
    if move == 0:
        bit = int(inactive[int(rng.integers(0, inactive.size))])
        return int(mask | (1 << bit))
    if move == 1:
        bit = int(active[int(rng.integers(0, active.size))])
        return int(mask & ~(1 << bit))

    remove_bit = int(active[int(rng.integers(0, active.size))])
    add_bit = int(inactive[int(rng.integers(0, inactive.size))])
    return int((mask & ~(1 << remove_bit)) | (1 << add_bit))


def local_search(data, prefix, max_lag=64, max_taps=16, restarts=10, steps=2000, seed=42):
    """
    Returns:
    best_taps (list)
    best_score (float)
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    data = np.asarray(data, dtype=np.uint8)
    prefix = np.asarray(prefix, dtype=np.uint8)

    if prefix.shape[0] < max_lag:
        max_lag = prefix.shape[0]

    best_mask = 0
    best_score = float("-inf")
    best_key = _score_key(best_mask, best_score)

    initial_masks = [0]
    initial_masks.extend(_heuristic_masks(data, prefix, max_lag=max_lag, max_taps=max_taps))
    while len(initial_masks) < restarts:
        initial_masks.append(_random_mask(rng, max_lag=max_lag, max_taps=max_taps))

    for restart_idx in range(restarts):
        current_mask = initial_masks[restart_idx]
        current_score = _score_mask(current_mask, data, prefix, noise=0.2, max_lag=max_lag)
        current_key = _score_key(current_mask, current_score)

        if current_key > best_key:
            best_mask = current_mask
            best_score = current_score
            best_key = current_key

        for step_idx in range(steps):
            candidate_mask = _propose_neighbor(current_mask, rng, max_lag=max_lag, max_taps=max_taps)
            candidate_score = _score_mask(candidate_mask, data, prefix, noise=0.2, max_lag=max_lag)
            candidate_key = _score_key(candidate_mask, candidate_score)

            if candidate_score == float("-inf"):
                continue

            temperature = max(1e-3, 1.0 - (step_idx / max(1, steps)))
            delta = candidate_score - current_score
            accept = candidate_key > current_key
            if not accept and current_score != float("-inf"):
                accept_prob = math.exp(delta / temperature)
                accept = rng.random() < accept_prob

            if accept:
                current_mask = candidate_mask
                current_score = candidate_score
                current_key = candidate_key

            if current_key > best_key:
                best_mask = current_mask
                best_score = current_score
                best_key = current_key

    best_taps = _mask_to_taps(best_mask, max_lag=max_lag)
    print(f"taps={best_taps}")
    print(f"score={best_score}")
    print(f"num_taps={len(best_taps)}")
    return best_taps, best_score


def optimized_local_search(data, prefix, max_lag=64, max_taps=16, restarts=10, steps=2000, seed=42, noise=0.2):
    """
    High-performance local search using uint64 bitmasks and vectorized parity scoring.

    Returns:
        best_mask (int), best_score (float)
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    data = np.asarray(data, dtype=np.uint8)
    prefix = np.asarray(prefix, dtype=np.uint8)
    max_lag = min(int(max_lag), 64, data.shape[1] - 1, prefix.shape[0])

    contexts, y = _build_lagged_bitmasks(data, max_lag=max_lag)
    prefix_contexts, prefix_targets = _build_prefix_bitmasks(prefix, max_lag=max_lag)

    best_mask = 0
    best_score = float("-inf")
    best_key = _score_key(best_mask, best_score)

    heuristic_masks = [mask for mask in _heuristic_masks(data, prefix, max_lag=max_lag, max_taps=max_taps) if _popcount(mask) <= max_taps]
    initial_masks = heuristic_masks[:]
    while len(initial_masks) < restarts:
        initial_masks.append(_random_mask(rng, max_lag=max_lag, max_taps=max_taps))

    for restart_idx in range(restarts):
        current_mask = int(initial_masks[restart_idx])
        current_score = _score_mask_fast(current_mask, contexts, y, prefix_contexts, prefix_targets, noise=noise)
        current_key = _score_key(current_mask, current_score)
        accepted_moves = 0
        progression = []

        if current_key > best_key:
            best_mask = current_mask
            best_score = current_score
            best_key = current_key

        for step_idx in range(steps):
            candidate_mask = _propose_neighbor_mask(current_mask, rng, max_lag=max_lag, max_taps=max_taps)
            candidate_score = _score_mask_fast(candidate_mask, contexts, y, prefix_contexts, prefix_targets, noise=noise)
            if candidate_score == float("-inf"):
                continue

            candidate_key = _score_key(candidate_mask, candidate_score)
            temperature = max(1e-6, 1.0 - (step_idx / max(1, steps)))
            delta = candidate_score - current_score
            accept = candidate_key > current_key
            if not accept and current_score != float("-inf"):
                accept = rng.random() < math.exp(delta / temperature)

            if accept:
                current_mask = candidate_mask
                current_score = candidate_score
                current_key = candidate_key
                accepted_moves += 1

            if current_key > best_key:
                best_mask = current_mask
                best_score = current_score
                best_key = current_key
                progression.append((step_idx, best_score, _popcount(best_mask)))

        print(
            "restart="
            f"{restart_idx + 1}/{restarts} "
            f"initial_taps={_mask_to_taps(initial_masks[restart_idx], max_lag=max_lag)} "
            f"accepted_moves={accepted_moves} "
            f"best_progression={progression[:8]} "
            f"final_taps={_mask_to_taps(current_mask, max_lag=max_lag)}"
        )

    best_taps = _mask_to_taps(best_mask, max_lag=max_lag)
    print(f"taps={best_taps}")
    print(f"score={best_score}")
    print(f"num_taps={len(best_taps)}")
    return int(best_mask), float(best_score)

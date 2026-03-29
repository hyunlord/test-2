import math
import warnings

import numpy as np

from .scoring import check_prefix_consistency


def vectorized_wht(data_01: np.ndarray, data_pm: np.ndarray, w: int, verbose=True) -> np.ndarray:
    """
    Compute Walsh coefficients for all 2^w masks via histogram + in-place WHT.
    """
    data_01 = np.asarray(data_01, dtype=np.uint8)
    data_pm = np.asarray(data_pm, dtype=np.float32)
    if data_01.shape != data_pm.shape:
        raise ValueError("data_01 and data_pm must have the same shape")
    if data_01.ndim != 2:
        raise ValueError("data arrays must be 2D")
    if not (1 <= w <= min(64, data_01.shape[1] - 1)):
        raise ValueError("w must be between 1 and 64 and smaller than sequence length")
    if w > 30:
        mem_gb = (1 << w) * 8 / 1e9
        warnings.warn(f"WHT with w={w} requires ~{mem_gb:.1f}GB memory")

    num_seq, seq_len = data_01.shape
    num_positions = seq_len - w
    packed = np.zeros((num_seq, num_positions), dtype=np.uint64)
    for d in range(w):
        packed |= data_01[:, w - 1 - d : seq_len - 1 - d].astype(np.uint64) << d

    hist_size = 1 << w
    H = np.zeros(hist_size, dtype=np.float64)
    targets = data_pm[:, w:]
    np.add.at(H, packed.ravel(), targets.ravel())

    step = 1
    while step < hist_size:
        stride = step * 2
        for start in range(0, hist_size, stride):
            x = H[start : start + step].copy()
            y = H[start + step : start + stride].copy()
            H[start : start + step] = x + y
            H[start + step : start + stride] = x - y
        step = stride

    total_samples = num_seq * num_positions
    H /= float(total_samples)
    if verbose:
        max_idx = int(np.argmax(np.abs(H)))
        print(f"[WHT] w={w} max_abs_bias={abs(H[max_idx]):.6f} mask={max_idx}")
    return H


def expected_noise_floor(w: int, N_seq: int, L_seq: int = 256) -> tuple[float, float]:
    sigma = 1.0 / math.sqrt(max(1, N_seq * (L_seq - w)))
    expected_max = sigma * math.sqrt(2.0 * math.log(max(2, 1 << w)))
    return sigma, expected_max


def wht_top_masks(F: np.ndarray, w: int, top_k: int = 10) -> list[dict]:
    F = np.asarray(F, dtype=np.float64)
    order = np.argsort(np.abs(F))[::-1][:top_k]
    results = []
    for idx in order:
        offsets = [bit + 1 for bit in range(w) if (int(idx) >> bit) & 1]
        results.append(
            {
                "mask": int(idx),
                "offsets": offsets,
                "bias": float(F[idx]),
                "abs_bias": float(abs(F[idx])),
                "S": len(offsets),
            }
        )
    return results


def run_wht_scan(data_01, data_pm, w_range=None, prefix=None, verbose=True) -> dict:
    if w_range is None:
        w_range = [8, 12, 16, 20, 24, 28, 32]
    prefix = None if prefix is None else np.asarray(prefix, dtype=np.uint8)

    scan_results = []
    best_offsets = None
    best_abs_bias = -1.0
    signal_found = False

    for w in w_range:
        if w >= data_01.shape[1]:
            continue
        F = vectorized_wht(data_01, data_pm, w=w, verbose=verbose)
        sigma, expected_max = expected_noise_floor(w=w, N_seq=data_01.shape[0], L_seq=data_01.shape[1])
        top_masks = wht_top_masks(F, w=w, top_k=10)

        valid_top = []
        for item in top_masks:
            if prefix is None:
                item["prefix_consistent"] = None
                valid_top.append(item)
            else:
                ok = bool(check_prefix_consistency(item["offsets"], prefix))
                item["prefix_consistent"] = ok
                if ok:
                    valid_top.append(item)

        candidate_pool = valid_top if valid_top else top_masks
        if candidate_pool:
            top_item = candidate_pool[0]
            if top_item["abs_bias"] > best_abs_bias:
                best_abs_bias = top_item["abs_bias"]
                best_offsets = top_item["offsets"]
            if top_item["abs_bias"] > 0.3:
                signal_found = True

        scan_results.append(
            {
                "w": int(w),
                "sigma": float(sigma),
                "expected_max": float(expected_max),
                "top_masks": top_masks,
                "best_valid_offsets": None if not candidate_pool else candidate_pool[0]["offsets"],
            }
        )
        if signal_found:
            break

    return {
        "scan_results": scan_results,
        "best_offsets": best_offsets,
        "signal_found": bool(signal_found),
    }

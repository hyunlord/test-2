import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def taps_to_mask(taps):
    mask = 0
    for tap in sorted(set(int(t) for t in taps)):
        mask |= 1 << (tap - 1)
    return mask


def mask_to_taps(mask, max_lag=64):
    mask = int(mask)
    return [lag for lag in range(1, max_lag + 1) if (mask >> (lag - 1)) & 1]


def rollout_sequence(prefix, taps, total_length=256):
    prefix = np.asarray(prefix, dtype=np.uint8)
    taps = sorted(set(int(t) for t in taps))
    generated = np.zeros(total_length, dtype=np.uint8)
    generated[: len(prefix)] = prefix
    if not taps:
        return generated
    for pos in range(len(prefix), total_length):
        value = 0
        for tap in taps:
            value ^= int(generated[pos - tap])
        generated[pos] = value
    return generated


def extract_answer_bits(sequence, start=64, end=256):
    sequence = np.asarray(sequence, dtype=np.uint8)
    return "".join(str(int(bit)) for bit in sequence[start:end])


def log_failed_attempt(path, method, config, result, note, promising=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "config": config,
        "result": result,
        "note": note,
        "promising": bool(promising),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return record

import random
from pathlib import Path

import numpy as np


DEFAULT_TEST_PREFIX = "0000010100011010010101100101001110100011110010110011010000111010"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def load_data(path):
    path = Path(path)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No sequences found in {path}")
    widths = {len(line) for line in lines}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent sequence lengths in {path}: {sorted(widths)}")
    if any(set(line) - {"0", "1"} for line in lines):
        raise ValueError(f"Non-binary character detected in {path}")
    data = np.array([[int(bit) for bit in line] for line in lines], dtype=np.uint8)
    return data


def get_test_prefix():
    return np.array([int(bit) for bit in DEFAULT_TEST_PREFIX], dtype=np.uint8)


def build_dataset(data, max_lag=64):
    data = np.asarray(data, dtype=np.uint8)
    if data.ndim != 2:
        raise ValueError("data must be a 2D array of binary sequences")
    num_sequences, seq_len = data.shape
    if max_lag <= 0 or max_lag >= seq_len:
        raise ValueError("max_lag must be between 1 and sequence length - 1")

    X = []
    y = []
    seq_ids = []
    positions = []
    for seq_idx in range(num_sequences):
        for pos in range(max_lag, seq_len):
            X.append(data[seq_idx, pos - max_lag : pos][::-1].astype(np.float32))
            y.append(data[seq_idx, pos])
            seq_ids.append(seq_idx)
            positions.append(pos)

    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.uint8),
        "seq_ids": np.asarray(seq_ids, dtype=np.int32),
        "positions": np.asarray(positions, dtype=np.int32),
        "max_lag": max_lag,
        "num_sequences": num_sequences,
        "seq_len": seq_len,
    }

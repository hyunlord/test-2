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


def split_dataset_by_sequence(data, holdout_fraction=0.2, seed=42):
    data = np.asarray(data, dtype=np.uint8)
    if data.ndim != 2:
        raise ValueError("data must be a 2D array")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = np.arange(data.shape[0])
    rng.shuffle(indices)
    holdout_size = max(1, int(round(data.shape[0] * holdout_fraction)))
    holdout_idx = np.sort(indices[:holdout_size])
    train_idx = np.sort(indices[holdout_size:])
    return data[train_idx], data[holdout_idx]


def temporal_train_validation_split(dataset, validation_fraction=0.2):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    X = np.asarray(dataset["X"])
    y = np.asarray(dataset["y"])
    positions = np.asarray(dataset["positions"])

    cutoff = np.quantile(positions, 1.0 - validation_fraction)
    train_mask = positions < cutoff
    valid_mask = ~train_mask

    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        midpoint = max(1, int(round(len(X) * (1.0 - validation_fraction))))
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[:midpoint] = True
        valid_mask = ~train_mask

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_valid": X[valid_mask],
        "y_valid": y[valid_mask],
        "positions_train": positions[train_mask],
        "positions_valid": positions[valid_mask],
    }

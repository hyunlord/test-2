import json
import math
import random
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from sparsetap_config import DEFAULT_CONFIG, SparseTapConfig, ensure_artifact_dirs
ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sparsetap.data import get_test_prefix as _get_test_prefix
from sparsetap.data import load_data
from sparsetap.models import run_logistic_l1 as _run_logistic_l1
from sparsetap.models import run_mlp as _run_mlp
from sparsetap.models import run_xgboost as _run_xgboost
from sparsetap.scoring import (
    check_prefix_consistency as _check_prefix_consistency,
    compute_accuracy as _compute_accuracy,
    compute_log_likelihood as _compute_log_likelihood,
    rank_candidates as _rank_candidates,
)
from sparsetap.search import (
    rollout_bits as _rollout_bits,
    run_beam_search as _run_beam_search,
    run_greedy_search as _run_greedy_search,
    run_local_search as _run_local_search,
    run_prefix_solver as _run_prefix_solver,
    run_reduced_exhaustive_search as _run_reduced_exhaustive_search,
)
from sparsetap.statistical import (
    run_pair_scan as _run_pair_scan,
    run_single_lag_scan as _run_single_lag_scan,
    summarize_tap_support,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def load_sequences(path: str) -> np.ndarray:
    return load_data(path)


def validate_sequences(arr: np.ndarray):
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array of binary sequences.")
    if arr.shape[1] != 256:
        raise ValueError(f"Expected sequence length 256, got {arr.shape[1]}.")
    unique_values = np.unique(arr)
    if np.setdiff1d(unique_values, np.array([0, 1])).size:
        raise ValueError(f"Expected binary values only, got {unique_values.tolist()}.")
    return {"num_sequences": int(arr.shape[0]), "sequence_length": int(arr.shape[1])}


def prefix_to_array(prefix: str) -> np.ndarray:
    prefix = prefix.strip()
    if len(prefix) != 64:
        raise ValueError(f"Expected 64-bit prefix, got {len(prefix)} bits.")
    if set(prefix) - {"0", "1"}:
        raise ValueError("Prefix must contain only '0' and '1'.")
    return np.array([int(bit) for bit in prefix], dtype=np.uint8)


def build_supervised_dataset(arr, max_lag=64):
    arr = np.asarray(arr, dtype=np.uint8)
    X = []
    y = []
    seq_ids = []
    positions = []
    for seq_idx in range(arr.shape[0]):
        for pos in range(max_lag, arr.shape[1]):
            X.append(arr[seq_idx, pos - max_lag : pos][::-1].astype(np.float32))
            y.append(arr[seq_idx, pos])
            seq_ids.append(seq_idx)
            positions.append(pos)
    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.uint8),
        "seq_ids": np.asarray(seq_ids, dtype=np.int32),
        "positions": np.asarray(positions, dtype=np.int32),
        "max_lag": max_lag,
    }


def check_prefix_consistency(taps, prefix):
    return _check_prefix_consistency(taps, prefix)


def compute_log_likelihood(taps, data, noise=0.2):
    return _compute_log_likelihood(taps, data, noise=noise)


def compute_accuracy(taps, data):
    return _compute_accuracy(taps, data)


def make_candidate(track, method, taps=None, W=None, scores=None, metadata=None, note=""):
    taps = sorted(set(taps or []))
    if W is None and taps:
        W = max(taps)
    return {
        "candidate_id": f"{track}-{method}-{uuid.uuid4().hex[:8]}",
        "track": track,
        "method": method,
        "taps": taps,
        "W": W,
        "scores": scores or {},
        "metadata": {"note": note, **(metadata or {})},
    }


def evaluate_rule_candidate(candidate, data, prefix, noise=0.2):
    taps = sorted(set(candidate.get("taps", [])))
    scores = dict(candidate.get("scores", {}))
    scores["prefix_consistent"] = int(_check_prefix_consistency(taps, prefix))
    scores["log_likelihood"] = float(_compute_log_likelihood(taps, data, noise=noise)) if taps else float("-inf")
    scores["accuracy"] = float(_compute_accuracy(taps, data)) if taps else 0.0
    scores["num_taps"] = len(taps)
    scores["rank_tuple"] = (
        scores["prefix_consistent"],
        scores["log_likelihood"],
        scores["accuracy"],
        -scores["num_taps"],
    )
    candidate["taps"] = taps
    candidate["W"] = max(taps) if taps else None
    candidate["scores"] = scores
    return candidate


def rank_candidates(candidates, data=None, prefix=None, noise=0.2):
    normalized = []
    for candidate in candidates:
        candidate = dict(candidate)
        if data is not None and prefix is not None and candidate.get("taps") is not None:
            candidate = evaluate_rule_candidate(candidate, data=data, prefix=prefix, noise=noise)
        else:
            scores = dict(candidate.get("scores", {}))
            num_taps = scores.get("num_taps", len(candidate.get("taps", []) or []))
            scores["num_taps"] = num_taps
            scores["rank_tuple"] = (
                scores.get("prefix_consistent", 0),
                scores.get("log_likelihood", float("-inf")),
                scores.get("accuracy", 0.0),
                -num_taps,
            )
            candidate["scores"] = scores
        normalized.append(candidate)
    normalized.sort(key=lambda item: item["scores"]["rank_tuple"], reverse=True)
    return normalized


def save_candidates(candidates, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidates, indent=2))
    return path


def load_candidates(path):
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def rollout_from_prefix(prefix, taps, total_len=256):
    return _rollout_bits(taps, prefix, total_length=total_len)


def extract_answer_bits(seq, start=64, end=256):
    return "".join(str(int(bit)) for bit in np.asarray(seq)[start:end])


def save_final_answer(answer, candidate, path_txt, path_json):
    path_txt = Path(path_txt)
    path_json = Path(path_json)
    path_txt.parent.mkdir(parents=True, exist_ok=True)
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_txt.write_text(answer + "\n")
    path_json.write_text(json.dumps(candidate, indent=2))
    return path_txt, path_json


def prepare_environment(config: SparseTapConfig = DEFAULT_CONFIG):
    set_seed(config.seed)
    ensure_artifact_dirs(config)
    data = load_sequences(str(config.data_path))
    validate_sequences(data)
    prefix = prefix_to_array(config.test_prefix)
    return data, prefix


def run_single_lag_scan(data, prefix, config: SparseTapConfig = DEFAULT_CONFIG):
    raw = _run_single_lag_scan(data, prefix, max_lag=config.max_lag, top_k=config.candidate_top_k, noise=config.noise_prob)
    candidates = [
        evaluate_rule_candidate(
            make_candidate(
                track="predictive",
                method="single_lag_scan",
                taps=item["taps"],
                metadata={"lag": item["taps"][0], "source_scores": item["scores"]},
                note="Simple one-lag agreement baseline.",
            ),
            data,
            prefix,
            noise=config.noise_prob,
        )
        for item in raw["candidates"]
    ]
    return {"records": raw["records"], "candidates": candidates}


def run_pair_scan(data, prefix, candidate_lags=None, config: SparseTapConfig = DEFAULT_CONFIG):
    raw = _run_pair_scan(
        data,
        prefix,
        max_lag=config.max_lag,
        top_k=config.candidate_top_k,
        top_lags=candidate_lags,
        noise=config.noise_prob,
    )
    candidates = [
        evaluate_rule_candidate(
            make_candidate(
                track="predictive",
                method="pair_scan",
                taps=item["taps"],
                metadata={"source_scores": item["scores"]},
                note="Pairwise XOR interaction screen.",
            ),
            data,
            prefix,
            noise=config.noise_prob,
        )
        for item in raw["candidates"]
    ]
    return {"records": raw["records"], "candidates": candidates}


def run_logistic_l1(dataset, data, prefix, config: SparseTapConfig = DEFAULT_CONFIG):
    raw = _run_logistic_l1(dataset, data, prefix, c_grid=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0], noise=config.noise_prob, max_taps=config.max_taps)
    return [
        evaluate_rule_candidate(
            make_candidate(
                track="predictive",
                method="logistic_l1",
                taps=item["taps"],
                metadata=item.get("metadata", {}),
                note="Sparse linear next-bit baseline; useful for lag priors.",
            ),
            data,
            prefix,
            noise=config.noise_prob,
        )
        for item in raw
    ]


def run_xgboost(dataset, data, prefix, config: SparseTapConfig = DEFAULT_CONFIG):
    raw = _run_xgboost(dataset, data, prefix, top_k=config.candidate_top_k, noise=config.noise_prob, max_taps=config.max_taps)
    return [
        evaluate_rule_candidate(
            make_candidate(
                track="predictive",
                method="xgboost",
                taps=item["taps"],
                metadata=item.get("metadata", {}),
                note="Tree-based predictive baseline converted into tap priors.",
            ),
            data,
            prefix,
            noise=config.noise_prob,
        )
        for item in raw
    ]


def run_mlp(dataset, data, prefix, config: SparseTapConfig = DEFAULT_CONFIG):
    raw = _run_mlp(dataset, data, prefix, hidden_dim=64, epochs=10, noise=config.noise_prob, max_taps=config.max_taps)
    return [
        evaluate_rule_candidate(
            make_candidate(
                track="predictive",
                method="mlp",
                taps=item["taps"],
                metadata=item.get("metadata", {}),
                note="Lightweight MLP baseline; mostly for comparison and failure analysis.",
            ),
            data,
            prefix,
            noise=config.noise_prob,
        )
        for item in raw
    ]


def run_prefix_solver(prefix, candidate_lags, config: SparseTapConfig = DEFAULT_CONFIG):
    raw = _run_prefix_solver(prefix, candidate_lags=candidate_lags, max_taps=min(config.max_taps, 6))
    return [
        make_candidate(
            track="rule_recovery",
            method="prefix_solver",
            taps=item["taps"],
            metadata={"prefix_only": True},
            note="Exact prefix feasibility search.",
        )
        for item in raw
    ]


def run_greedy_search(candidate_pool, data, prefix, config: SparseTapConfig = DEFAULT_CONFIG):
    raw = _run_greedy_search(candidate_pool, data, prefix, noise=config.noise_prob, max_taps=config.max_taps)
    return raw


def run_beam_search(candidate_pool, data, prefix, config: SparseTapConfig = DEFAULT_CONFIG):
    return _run_beam_search(candidate_pool, data, prefix, noise=config.noise_prob, beam_width=config.beam_width, max_taps=config.max_taps)


def run_local_search(seed_taps, candidate_pool, data, prefix, config: SparseTapConfig = DEFAULT_CONFIG):
    return _run_local_search(seed_taps, candidate_pool, data, prefix, noise=config.noise_prob, max_taps=config.max_taps, max_rounds=config.local_search_steps)


def run_reduced_exhaustive_search(candidate_pool, data, prefix, config: SparseTapConfig = DEFAULT_CONFIG, max_combination_size=5):
    return _run_reduced_exhaustive_search(candidate_pool, data, prefix, noise=config.noise_prob, max_combination_size=max_combination_size)


def build_rule_recovery_pool(stat_candidates, ml_candidates, config: SparseTapConfig = DEFAULT_CONFIG):
    merged = stat_candidates + ml_candidates
    support = summarize_tap_support(merged, max_lag=config.max_lag)
    pool = [idx for idx in np.argsort(support)[::-1] if idx > 0][: max(12, min(24, config.max_lag))]
    return pool, support


def candidate_table(candidates):
    rows = []
    for cand in candidates:
        rows.append(
            {
                "candidate_id": cand.get("candidate_id"),
                "track": cand.get("track"),
                "method": cand.get("method"),
                "taps": cand.get("taps"),
                "W": cand.get("W"),
                "prefix consistent?": cand.get("scores", {}).get("prefix_consistent"),
                "log-likelihood": cand.get("scores", {}).get("log_likelihood"),
                "accuracy": cand.get("scores", {}).get("accuracy"),
                "number of taps": cand.get("scores", {}).get("num_taps"),
                "notes": cand.get("metadata", {}).get("note", ""),
            }
        )
    return pd.DataFrame(rows)


def get_default_prefix_array():
    return _get_test_prefix()

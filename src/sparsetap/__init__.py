from .data import build_dataset, load_data, set_seed
from .data import split_dataset_by_sequence, temporal_train_validation_split
from .evaluation import evaluate_candidate, rank_candidates_table
from .io import load_candidates, save_candidates, save_results
from .models import run_logistic_l1, run_mlp, run_xgboost
from .scoring import (
    check_prefix_consistency,
    compute_accuracy,
    compute_log_likelihood,
    rank_candidates,
    score_candidate,
)
from .search import (
    run_greedy_search,
    run_reduced_exhaustive_search,
    rollout_bits,
    run_beam_search,
    run_local_search,
    run_prefix_solver,
)
from .statistical import run_pair_scan, run_single_lag_scan
from .utils import extract_answer_bits, log_failed_attempt, mask_to_taps, rollout_sequence, taps_to_mask

__all__ = [
    "build_dataset",
    "evaluate_candidate",
    "check_prefix_consistency",
    "compute_accuracy",
    "compute_log_likelihood",
    "extract_answer_bits",
    "load_candidates",
    "load_data",
    "log_failed_attempt",
    "mask_to_taps",
    "rank_candidates",
    "rank_candidates_table",
    "rollout_bits",
    "rollout_sequence",
    "run_beam_search",
    "run_greedy_search",
    "run_local_search",
    "run_logistic_l1",
    "run_mlp",
    "run_pair_scan",
    "run_prefix_solver",
    "run_reduced_exhaustive_search",
    "run_single_lag_scan",
    "run_xgboost",
    "save_candidates",
    "save_results",
    "score_candidate",
    "set_seed",
    "split_dataset_by_sequence",
    "taps_to_mask",
    "temporal_train_validation_split",
]

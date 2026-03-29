import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def write_notebook(path, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=2))


BOOTSTRAP = """from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

SEARCH_ROOTS = [Path.cwd().resolve(), Path.cwd().resolve().parent]
PROJECT_ROOT = next(path for path in SEARCH_ROOTS if (path / "DAY2_data.txt").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
"""


def main():
    write_notebook(
        NOTEBOOK_DIR / "00_setup.ipynb",
        [
            md_cell(
                "# 00 Setup\n\n"
                "This notebook creates a deterministic SparseTap experiment environment, validates the DGX/Jupyter runtime, "
                "loads the training data, and creates the artifact directory structure used by the rest of the pipeline."
            ),
            code_cell(
                BOOTSTRAP
                + """
from sparsetap_config import DEFAULT_CONFIG, ensure_artifact_dirs
from sparsetap_utils import prepare_environment, validate_sequences

config = DEFAULT_CONFIG
ensure_artifact_dirs(config)
data, prefix = prepare_environment(config)
summary = validate_sequences(data)
summary["prefix_length"] = int(len(prefix))
summary["artifact_root"] = str(config.artifact_root)
summary["seed"] = config.seed
summary
"""
            ),
            code_cell(
                BOOTSTRAP
                + """
from sparsetap_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG
config_dict = {
    "seed": config.seed,
    "max_lag": config.max_lag,
    "noise_prob": config.noise_prob,
    "max_taps": config.max_taps,
    "beam_width": config.beam_width,
    "local_search_restarts": config.local_search_restarts,
    "local_search_steps": config.local_search_steps,
    "candidate_top_k": config.candidate_top_k,
    "data_path": str(config.data_path),
    "artifact_root": str(config.artifact_root),
    "test_prefix": config.test_prefix,
}
(config.artifact_root / "config_snapshot.json").write_text(json.dumps(config_dict, indent=2))
print(json.dumps(config_dict, indent=2))
"""
            ),
        ],
    )

    write_notebook(
        NOTEBOOK_DIR / "01_data_inspection.ipynb",
        [
            md_cell(
                "# 01 Data Inspection\n\n"
                "This notebook inspects the raw binary corpus before modeling. It is designed to support report writing: "
                "shape checks, bit balance, lag agreement summaries, and a quick sanity view of the test prefix."
            ),
            code_cell(
                BOOTSTRAP
                + """
from sparsetap_config import DEFAULT_CONFIG
from sparsetap_utils import prepare_environment

config = DEFAULT_CONFIG
data, prefix = prepare_environment(config)

overview = {
    "num_sequences": int(data.shape[0]),
    "sequence_length": int(data.shape[1]),
    "overall_bit_mean": float(data.mean()),
    "prefix_bit_mean": float(prefix.mean()),
}
overview
"""
            ),
            code_cell(
                BOOTSTRAP
                + """
from sparsetap_config import DEFAULT_CONFIG
from sparsetap_utils import prepare_environment

config = DEFAULT_CONFIG
data, prefix = prepare_environment(config)

position_mean = pd.Series(data.mean(axis=0), name="bit_mean")
lag_agreement = pd.DataFrame(
    [
        {
            "lag": lag,
            "agreement": float((data[:, lag:] == data[:, :-lag]).mean()),
        }
        for lag in range(1, config.max_lag + 1)
    ]
)

display(position_mean.head(16).to_frame())
display(lag_agreement.sort_values("agreement", ascending=False).head(15))
print("Test prefix:", "".join(str(int(bit)) for bit in prefix))
"""
            ),
        ],
    )

    write_notebook(
        NOTEBOOK_DIR / "02_predictive_track.ipynb",
        [
            md_cell(
                "# 02 Predictive Track\n\n"
                "Track A treats SparseTap as next-bit prediction. We run statistical lag scans, Logistic Regression with L1, "
                "XGBoost or fallback boosting, and a lightweight MLP. Every method saves artifacts and candidate rule hints."
            ),
            code_cell(
                BOOTSTRAP
                + """
from sparsetap_config import DEFAULT_CONFIG
from sparsetap_utils import (
    build_supervised_dataset,
    candidate_table,
    prepare_environment,
    run_logistic_l1,
    run_mlp,
    run_pair_scan,
    run_single_lag_scan,
    run_xgboost,
    save_candidates,
    save_table,
)

config = DEFAULT_CONFIG
data, prefix = prepare_environment(config)
dataset = build_supervised_dataset(data, max_lag=config.max_lag)

single = run_single_lag_scan(data, prefix, config)
pair = run_pair_scan(data, prefix, [cand["taps"][0] for cand in single["candidates"][:12]], config)
logistic = run_logistic_l1(dataset, data, prefix, config)
xgb = run_xgboost(dataset, data, prefix, config)
mlp = run_mlp(dataset, data, prefix, config)

predictive_candidates = single["candidates"] + pair["candidates"] + logistic + xgb + mlp
save_candidates(predictive_candidates, config.candidate_dir / "predictive_candidates.json")
predictive_table = candidate_table(predictive_candidates).sort_values(
    ["prefix consistent?", "log-likelihood", "accuracy", "number of taps"],
    ascending=[False, False, False, True],
)
save_table(predictive_table, config.metrics_dir / "predictive_summary.csv")
predictive_table.head(30)
"""
            ),
            md_cell(
                "This notebook also supports failure analysis. Predictive methods store notes in metadata so the final report can summarize "
                "which baselines were weak, which were only useful as priors, and why raw predictive accuracy alone is not enough."
            ),
        ],
    )

    write_notebook(
        NOTEBOOK_DIR / "03_rule_recovery_track.ipynb",
        [
            md_cell(
                "# 03 Rule Recovery Track\n\n"
                "Track B searches directly for the hidden XOR tap set. It consumes Track A outputs as priors, applies clean-prefix filtering, "
                "and ranks candidates with noise-aware log-likelihood and training accuracy."
            ),
            code_cell(
                BOOTSTRAP
                + """
from sparsetap_config import DEFAULT_CONFIG
from sparsetap_utils import (
    build_rule_recovery_pool,
    candidate_table,
    evaluate_rule_candidate,
    load_candidates,
    prepare_environment,
    rank_candidates,
    run_beam_search,
    run_greedy_search,
    run_local_search,
    run_prefix_solver,
    run_reduced_exhaustive_search,
    save_candidates,
    save_table,
)

config = DEFAULT_CONFIG
data, prefix = prepare_environment(config)
predictive_candidates = load_candidates(config.candidate_dir / "predictive_candidates.json")

candidate_pool, support = build_rule_recovery_pool(predictive_candidates[: config.candidate_top_k], [], config)
prefix_only = [evaluate_rule_candidate(cand, data, prefix, config.noise_prob) for cand in run_prefix_solver(prefix, candidate_pool[:12], config)]
greedy = run_greedy_search(candidate_pool[:16], data, prefix, config)
beam = run_beam_search(candidate_pool[:16], data, prefix, config)
local_runs = []
for restart_idx, seed_candidate in enumerate(beam["candidates"][: config.local_search_restarts], start=1):
    local_result = run_local_search(seed_candidate["taps"], candidate_pool[:16], data, prefix, config)
    best_local = dict(local_result["best"])
    best_local["track"] = "rule_recovery"
    best_local["metadata"] = {**best_local.get("metadata", {}), "restart": restart_idx, "note": "Local search refinement from beam seed."}
    local_runs.append(best_local)
reduced = run_reduced_exhaustive_search(candidate_pool[:10], data, prefix, config, max_combination_size=4)

rule_candidates = rank_candidates(
    prefix_only
    + beam["candidates"][: config.candidate_top_k]
    + greedy["history"][: config.candidate_top_k]
    + local_runs
    + reduced[: config.candidate_top_k],
    data=data,
    prefix=prefix,
    noise=config.noise_prob,
)

save_candidates(rule_candidates, config.candidate_dir / "rule_recovery_candidates.json")
rule_table = candidate_table(rule_candidates)
save_table(rule_table, config.metrics_dir / "rule_recovery_summary.csv")
print("Candidate pool:", candidate_pool[:16])
rule_table.head(30)
"""
            ),
            md_cell(
                "The clean 64-bit prefix acts as a hard compatibility test here. That makes it easy later to explain why some high-likelihood "
                "noisy candidates still fail to be viable final rules."
            ),
        ],
    )

    write_notebook(
        NOTEBOOK_DIR / "04_candidate_evaluation.ipynb",
        [
            md_cell(
                "# 04 Candidate Evaluation\n\n"
                "Track C merges outputs from both experimental views. The goal is a clean, report-friendly comparison table with shared scoring, "
                "method notes, and a single ranking rule."
            ),
            code_cell(
                BOOTSTRAP
                + """
from sparsetap_config import DEFAULT_CONFIG
from sparsetap_utils import (
    candidate_table,
    load_candidates,
    prepare_environment,
    rank_candidates,
    save_candidates,
    save_table,
)

config = DEFAULT_CONFIG
data, prefix = prepare_environment(config)

predictive_candidates = load_candidates(config.candidate_dir / "predictive_candidates.json")
rule_candidates = load_candidates(config.candidate_dir / "rule_recovery_candidates.json")
all_candidates = predictive_candidates + rule_candidates

ranked = rank_candidates(all_candidates, data=data, prefix=prefix, noise=config.noise_prob)
save_candidates(ranked, config.candidate_dir / "all_ranked_candidates.json")
ranking_table = candidate_table(ranked)
save_table(ranking_table, config.metrics_dir / "all_candidate_ranking.csv")

by_method = (
    ranking_table.sort_values(
        ["prefix consistent?", "log-likelihood", "accuracy", "number of taps"],
        ascending=[False, False, False, True],
    )
    .groupby(["track", "method"], as_index=False)
    .head(1)
)
display(by_method)
ranking_table.head(40)
"""
            ),
        ],
    )

    write_notebook(
        NOTEBOOK_DIR / "05_final_prediction.ipynb",
        [
            md_cell(
                "# 05 Final Prediction\n\n"
                "This notebook selects the best ranked candidate, rolls the clean prefix forward deterministically, extracts the 192 answer bits, "
                "and saves both the final answer and best candidate for submission packaging."
            ),
            code_cell(
                BOOTSTRAP
                + """
from sparsetap_config import DEFAULT_CONFIG
from sparsetap_utils import (
    extract_answer_bits,
    load_candidates,
    prepare_environment,
    rank_candidates,
    rollout_from_prefix,
    save_final_answer,
)

config = DEFAULT_CONFIG
data, prefix = prepare_environment(config)
ranked = load_candidates(config.candidate_dir / "all_ranked_candidates.json")
best = rank_candidates(ranked, data=data, prefix=prefix, noise=config.noise_prob)[0]

rolled = rollout_from_prefix(prefix, best["taps"], total_len=256)
answer = extract_answer_bits(rolled, start=64, end=256)

save_final_answer(
    answer,
    best,
    config.final_dir / "final_answer.txt",
    config.final_dir / "best_candidate.json",
)

print("Best tap set:", best["taps"])
print("Final 192-bit prediction string:")
print(answer)
best
"""
            ),
        ],
    )


if __name__ == "__main__":
    main()

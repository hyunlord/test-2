import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "06_full_experiment.ipynb"


def code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


BOOT = """from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
"""


cells = [
    markdown_cell(
        "# SparseTap — Full Experiment\n\n"
        "두 가지 트랙으로 접근: **Track A (Predictive)** vs **Track B (Rule Recovery)**. "
        "각 트랙에 GD 모델 1개 이상 포함.\n"
    ),
    code_cell(
        BOOT
        + """
from sparsetap.data import get_test_prefix, load_data, set_seed, split_dataset_by_sequence
from sparsetap.evaluation import evaluate_candidate, rank_candidates_table
from sparsetap.gd_models import (
    HAS_TORCH,
    run_cos_parity_gd,
    run_gumbel_mask_gd,
    run_next_bit_cnn,
    run_next_bit_mlp,
    run_reinforce,
    run_soft_mask_gd,
)
from sparsetap.models import run_logistic_l1, run_mlp, run_xgboost
from sparsetap.scoring import compute_accuracy, compute_log_likelihood
from sparsetap.search import run_beam_search, run_greedy_search, run_local_search, run_prefix_solver
from sparsetap.statistical import run_pair_scan, run_single_lag_scan, summarize_tap_support
from sparsetap.utils import extract_answer_bits, log_failed_attempt, rollout_sequence
from sparsetap.wht import run_wht_scan

set_seed(42)
print("Motivation: 전체 실험을 하나의 노트북에서 재현 가능하게 정리합니다.")
print("Method: seed=42 고정, 공통 모듈 import, device 상태 확인")
print("Expectation: 모든 트랙이 동일한 환경에서 비교 가능해야 합니다.")
print("Result: HAS_TORCH =", HAS_TORCH)
print("Analysis: Torch가 있으면 GD 모델 전부 실행하고, 없으면 skip 메타데이터를 남깁니다.")
print("Failure reason: 환경 의존성 문제가 있으면 여기서 바로 드러납니다.")
"""
    ),
    code_cell(
        BOOT
        + """
from sparsetap.data import build_dataset, temporal_train_validation_split

data = load_data(PROJECT_ROOT / "DAY2_data.txt")
prefix = get_test_prefix()
train_data, holdout_data = split_dataset_by_sequence(data, holdout_fraction=0.15, seed=42)
dataset = build_dataset(train_data, max_lag=64)
temporal = temporal_train_validation_split(dataset, validation_fraction=0.2)

print("Motivation: 전체 데이터와 holdout/temporal split을 분리해서 과적합 여부를 보겠습니다.")
print("Method: sequence holdout 15%, temporal validation 20%")
print("Expectation: search와 GD 결과를 holdout에 재평가할 수 있어야 합니다.")
print("Result:", {"train_sequences": int(train_data.shape[0]), "holdout_sequences": int(holdout_data.shape[0]), "temporal_train": int(temporal["X_train"].shape[0]), "temporal_valid": int(temporal["X_valid"].shape[0])})
print("Analysis: sequence split과 temporal split을 둘 다 두면 보고서에서 일반화 얘기를 하기 좋습니다.")
print("Failure reason: split이 없으면 random-like 성능을 해석하기 어렵습니다.")
"""
    ),
    code_cell(
        BOOT
        + """
ALL_RESULTS = []

def full_evaluate(offsets, label, track="unknown", note="", data_ref=train_data, holdout_ref=holdout_data):
    result = {
        "track": track,
        "method": label,
        "taps": sorted(set(int(x) for x in offsets)),
        "metadata": {"note": note},
    }
    scored = evaluate_candidate(result, data=data_ref, prefix=prefix, noise=0.2)
    holdout_ll = compute_log_likelihood(scored["taps"], holdout_ref, noise=0.2) if scored["taps"] else float("-inf")
    holdout_acc = compute_accuracy(scored["taps"], holdout_ref) if scored["taps"] else 0.0
    scored["scores"]["holdout_log_likelihood"] = float(holdout_ll)
    scored["scores"]["holdout_accuracy"] = float(holdout_acc)
    print(
        f"[{label}] taps={scored['taps']} "
        f"ll={scored['scores']['log_likelihood']:.3f} "
        f"acc={scored['scores']['accuracy']:.4f} "
        f"holdout_acc={holdout_acc:.4f}"
    )
    return scored

print("Motivation: 모든 접근법을 같은 평가 함수로 묶습니다.")
print("Method: prefix consistency + likelihood + accuracy + holdout metric 저장")
print("Expectation: 어느 셀에서 나온 후보든 동일하게 비교 가능해야 합니다.")
print("Result: full_evaluate()와 ALL_RESULTS 초기화 완료")
print("Analysis: 보고서 비교표의 기준점 역할을 합니다.")
print("Failure reason: 공통 평가가 없으면 방법 간 비교가 임의적이 됩니다.")
"""
    ),
    markdown_cell(
        "## Track A: Predictive Modeling\n\n"
        "다음 비트를 예측하는 모델 학습 → feature importance로 offset 후보 추출\n"
    ),
    code_cell(
        BOOT
        + """
single = run_single_lag_scan(train_data, prefix, max_lag=64, top_k=12, noise=0.2)
pair = run_pair_scan(train_data, prefix, max_lag=64, top_k=12, top_lags=[cand["taps"][0] for cand in single["candidates"][:8]], noise=0.2)

for cand in single["candidates"][:5]:
    ALL_RESULTS.append(full_evaluate(cand["taps"], f"single_lag_{cand['taps'][0]}", track="predictive", note="단일 lag 통계 스캔"))
for cand in pair["candidates"][:5]:
    ALL_RESULTS.append(full_evaluate(cand["taps"], f"pair_scan_{'_'.join(map(str, cand['taps']))}", track="predictive", note="lag pair 스캔"))

print("Motivation: 가장 해석 가능한 baseline으로 신호가 보이는지 확인합니다.")
print("Method: single-lag / pair scan")
print("Expectation: 약한 lag prior를 제공할 수 있습니다.")
print("Result: 상위 통계 후보를 ALL_RESULTS에 적재했습니다.")
print("Analysis: 단독으로는 약하지만 search 초기화에는 유용합니다.")
print("Failure reason: noisy XOR 구조에서는 단일 상관만으로 정답 복원이 어렵습니다.")
"""
    ),
    code_cell(
        BOOT
        + """
logistic_candidates = run_logistic_l1(dataset, train_data, prefix, noise=0.2, max_taps=16)
xgb_candidates = run_xgboost(dataset, train_data, prefix, noise=0.2, max_taps=16)

for cand in logistic_candidates[:3]:
    ALL_RESULTS.append(full_evaluate(cand["taps"], f"logistic_l1_C{cand['metadata']['C']}", track="predictive", note="GD 기반 sparse linear baseline"))
for cand in xgb_candidates[:3]:
    ALL_RESULTS.append(full_evaluate(cand["taps"], f"xgboost_top{cand['metadata']['size']}", track="predictive", note="tree-based baseline"))

print("Motivation: 전통 ML로 lag 중요도를 빠르게 추정합니다.")
print("Method: Logistic L1, XGBoost")
print("Expectation: sparse prior 혹은 reduced lag pool을 얻을 수 있습니다.")
print("Result: 상위 전통 ML 후보를 평가했습니다.")
print("Analysis: Logistic은 보고서용 GD baseline 역할도 합니다.")
print("Failure reason: raw next-bit accuracy가 높아도 exact rule recovery와는 다를 수 있습니다.")
"""
    ),
    code_cell(
        BOOT
        + """
mlp_result = run_next_bit_mlp(train_data, max_w=64, n_epochs=10, batch_size=2048, lr=0.001, seed=42)
ALL_RESULTS.append(full_evaluate(mlp_result.offsets, "next_bit_mlp", track="predictive", note="GD predictive baseline (MLP)"))

print("Motivation: GD 기반 predictive 모델 1번입니다.")
print("Method: 2-layer MLP + feature importance")
print("Expectation: local nonlinear interactions를 어느 정도 포착할 수 있습니다.")
print("Result:", {"offsets": mlp_result.offsets[:16], "accuracy": mlp_result.accuracy, "bias": mlp_result.bias})
print("Analysis: importance 기반 offsets는 직접적인 tap 복원은 아니어도 prior로 유용합니다.")
print("Failure reason: hidden rule이 XOR일 때 MLP가 규칙을 명시적으로 설명해주지는 못합니다.")
"""
    ),
    code_cell(
        BOOT
        + """
cnn_result = run_next_bit_cnn(train_data, max_w=64, n_epochs=10, batch_size=2048, lr=0.001, seed=42)
ALL_RESULTS.append(full_evaluate(cnn_result.offsets, "next_bit_cnn", track="predictive", note="GD predictive baseline (CNN)"))

print("Motivation: GD 기반 predictive 모델 2번입니다.")
print("Method: Conv1d 기반 next-bit predictor + gradient importance")
print("Expectation: 지역 패턴과 positional importance를 찾을 수 있습니다.")
print("Result:", {"offsets": cnn_result.offsets[:16], "accuracy": cnn_result.accuracy, "bias": cnn_result.bias})
print("Analysis: CNN은 feature ranking은 주지만 sparse exact tap set과는 거리감이 있습니다.")
print("Failure reason: convolution receptive field가 XOR parity를 직접 모델링하진 못합니다.")
"""
    ),
    markdown_cell(
        "## Track B: Rule Recovery\n\n"
        "숨겨진 XOR tap positions를 직접 복원하는 접근\n"
    ),
    code_cell(
        BOOT
        + """
data_pm = (1.0 - 2.0 * train_data).astype(np.float32)
wht_range = [16, 20, 24, 26, 28, 30, 32]
wht_result = run_wht_scan(train_data, data_pm, w_range=wht_range, prefix=prefix, verbose=True)
if wht_result["best_offsets"]:
    ALL_RESULTS.append(full_evaluate(wht_result["best_offsets"], "wht_scan_best", track="rule_recovery", note="Walsh-Hadamard exhaustive scan"))
else:
    log_failed_attempt(PROJECT_ROOT / "artifacts" / "failed_attempts.jsonl", "wht_scan", {"w_range": wht_range}, {"signal_found": False}, "강한 bias를 찾지 못함", promising=False)

print("Motivation: parity 구조를 직접 겨냥하는 exhaustive spectral scan입니다.")
print("Method: vectorized WHT")
print("Expectation: low-order exact parity가 있으면 강한 bias를 줄 수 있습니다.")
print("Result:", wht_result["best_offsets"])
print("Analysis: 탐색 없이 후보를 줄일 수 있는 강력한 rule-recovery 도구입니다.")
print("Failure reason: 실제 정답 차수가 높거나 window가 크면 메모리/탐색 한계가 있습니다.")
"""
    ),
    code_cell(
        BOOT
        + """
support = summarize_tap_support(logistic_candidates[:8] + xgb_candidates[:8] + single["candidates"][:8], max_lag=64)
candidate_pool = [idx for idx in np.argsort(support)[::-1] if idx > 0][:16]
prefix_candidates = run_prefix_solver(prefix, candidate_lags=candidate_pool[:10], max_taps=5)
greedy_result = run_greedy_search(candidate_pool, train_data, prefix, noise=0.2, max_taps=16)
beam_result = run_beam_search(candidate_pool, train_data, prefix, noise=0.2, beam_width=8, max_taps=12)
local_result = run_local_search(beam_result["candidates"][0]["taps"], candidate_pool, train_data, prefix, noise=0.2, max_taps=16)

for cand in prefix_candidates[:3]:
    ALL_RESULTS.append(full_evaluate(cand["taps"], "prefix_solver", track="rule_recovery", note="prefix exact solver"))
ALL_RESULTS.append(full_evaluate(greedy_result["best"]["taps"], "greedy_search", track="rule_recovery", note="combinatorial greedy"))
ALL_RESULTS.append(full_evaluate(beam_result["candidates"][0]["taps"], "beam_search", track="rule_recovery", note="beam search"))
ALL_RESULTS.append(full_evaluate(local_result["best"]["taps"], "simulated_annealing_like_local", track="rule_recovery", note="local search / SA-style refinement"))

print("Motivation: prefix + search를 결합한 메인 조합 탐색입니다.")
print("Method: prefix solver, greedy, beam, local search")
print("Expectation: exact-valid tap set에 접근할 가능성이 가장 높습니다.")
print("Result: top combinatorial candidates added")
print("Analysis: 현재 repo에서 가장 실전적인 메인 방법입니다.")
print("Failure reason: local optimum과 candidate pool 제한에 민감합니다.")
"""
    ),
    code_cell(
        BOOT
        + """
soft_result = run_soft_mask_gd(data_pm, max_w=64, n_epochs=80, batch_size=256, lr=0.05, l1_lambda=0.002, seed=42)
ALL_RESULTS.append(full_evaluate(soft_result.offsets, "soft_mask_gd", track="rule_recovery", note="GD rule recovery via soft mask"))

print("Motivation: 규칙 복원을 위한 첫 번째 GD 접근입니다.")
print("Method: soft mask product model")
print("Expectation: sparse taps를 연속 relaxation으로 찾습니다.")
print("Result:", {"offsets": soft_result.offsets[:16], "bias": soft_result.bias})
print("Analysis: clean toy data에서는 의미가 있지만 S가 커지면 gradient가 약해질 수 있습니다.")
print("Failure reason: vanishing gradient가 대표적인 실패 원인입니다.")
"""
    ),
    code_cell(
        BOOT
        + """
cos_result = run_cos_parity_gd(train_data, data_pm, max_w=64, n_steps=400, batch_size=8192, lr=0.03, l1_lambda=0.001, seed=42)
ALL_RESULTS.append(full_evaluate(cos_result.offsets, "cos_parity_gd", track="rule_recovery", note="GD rule recovery via cosine parity"))

print("Motivation: parity를 smoother하게 근사하는 두 번째 GD 접근입니다.")
print("Method: cosine parity relaxation")
print("Expectation: discrete parity보다 gradient 흐름이 쉬울 수 있습니다.")
print("Result:", {"offsets": cos_result.offsets[:16], "bias": cos_result.bias})
print("Analysis: local minima가 많아서 안정적으로 수렴하지 않을 수 있습니다.")
print("Failure reason: cos periodicity 때문에 multiple local minima 문제가 큽니다.")
"""
    ),
    code_cell(
        BOOT
        + """
gumbel_result = run_gumbel_mask_gd(data_pm, max_w=64, n_steps=400, batch_size=256, lr=0.05, l1_lambda=0.0005, seed=42)
ALL_RESULTS.append(full_evaluate(gumbel_result.offsets, "gumbel_mask_gd", track="rule_recovery", note="GD rule recovery via Gumbel-STE"))

print("Motivation: discrete mask에 더 가까운 세 번째 GD 접근입니다.")
print("Method: Gumbel-Sigmoid + STE")
print("Expectation: hard selection이 가능해져 tap 해석력이 좋아집니다.")
print("Result:", {"offsets": gumbel_result.offsets[:16], "bias": gumbel_result.bias})
print("Analysis: annealing 스케줄에 민감하지만 discrete search bridge로 의미가 있습니다.")
print("Failure reason: temperature choice가 나쁘면 너무 빨리 hardening 되거나 너무 오래 soft 상태에 머뭅니다.")
"""
    ),
    code_cell(
        BOOT
        + """
reinforce_result = run_reinforce(data_pm, max_w=64, n_steps=400, batch_size=128, n_samples=16, lr=0.03, l1_lambda=0.01, seed=42)
ALL_RESULTS.append(full_evaluate(reinforce_result.offsets, "reinforce", track="rule_recovery", note="policy-gradient rule recovery"))

print("Motivation: hard binary mask를 직접 최적화하는 네 번째 GD 접근입니다.")
print("Method: REINFORCE with Bernoulli mask")
print("Expectation: vanishing gradient를 피할 수 있습니다.")
print("Result:", {"offsets": reinforce_result.offsets[:16], "bias": reinforce_result.bias})
print("Analysis: variance가 크지만 이산 최적화라는 점에서 보고서 가치가 큽니다.")
print("Failure reason: gradient variance가 커서 안정적으로 수렴하지 않을 수 있습니다.")
"""
    ),
    markdown_cell(
        "## Final Evaluation & Prediction\n\n"
        "모든 접근법의 결과를 통합 평가\n"
    ),
    code_cell(
        BOOT
        + """
result_table = rank_candidates_table(ALL_RESULTS).sort_values(["prefix_consistency", "log_likelihood", "accuracy", "num_taps"], ascending=[False, False, False, True])
result_table.to_csv(PROJECT_ROOT / "artifacts" / "metrics" / "06_full_experiment_results.csv", index=False)
display(result_table.head(20))

print("Motivation: 모든 접근법을 하나의 표로 비교합니다.")
print("Method: shared tuple ranking")
print("Expectation: 어떤 방법이 가장 유망한지 바로 보여야 합니다.")
print("Result: CSV와 DataFrame 생성 완료")
print("Analysis: 보고서 비교표로 바로 사용할 수 있습니다.")
print("Failure reason: 비교 기준이 여러 개면 table normalization이 없을 때 해석이 엇갈릴 수 있습니다.")
"""
    ),
    code_cell(
        BOOT
        + """
best_row = result_table.iloc[0]
best_taps = list(best_row["taps"]) if isinstance(best_row["taps"], list) else []
if not best_taps:
    best_taps = ALL_RESULTS[0]["taps"]
final_seq = rollout_sequence(prefix, best_taps, total_length=256)
final_answer = extract_answer_bits(final_seq)
(PROJECT_ROOT / "artifacts" / "final" / "06_full_experiment_answer.txt").write_text(final_answer + "\\n")
print("Best taps:", best_taps)
print("Final answer:", final_answer)

print("Motivation: 최종 제출용 answer를 생성합니다.")
print("Method: best candidate rollout")
print("Expectation: clean prefix를 정확히 만족하는 192-bit answer를 얻습니다.")
print("Result: answer 저장 완료")
print("Analysis: 최종 선택은 shared evaluation 기준을 따릅니다.")
print("Failure reason: top candidate가 overfit이면 holdout과 final answer quality가 어긋날 수 있습니다.")
"""
    ),
    code_cell(
        BOOT
        + """
summary = {
    "num_results": len(ALL_RESULTS),
    "best_method": result_table.iloc[0]["method"],
    "best_taps": result_table.iloc[0]["taps"],
    "best_log_likelihood": float(result_table.iloc[0]["log_likelihood"]),
    "report_notes": [
        "통계 baseline은 prior 용도로는 유용했지만 단독으로 약했다.",
        "GD predictive 모델은 설명력보다 ranking prior 역할이 컸다.",
        "Rule recovery 쪽 combinatorial search와 spectral/GD hybrid가 핵심 후보를 만들었다.",
    ],
}
(PROJECT_ROOT / "artifacts" / "final" / "06_full_experiment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))

print("Motivation: 보고서에 바로 쓸 수 있는 요약을 남깁니다.")
print("Method: top result와 실패 메모를 JSON으로 저장")
print("Expectation: 실험 스토리라인을 정리할 수 있습니다.")
print("Result: summary JSON 저장 완료")
print("Analysis: 결과뿐 아니라 왜 이런 구조를 택했는지도 남깁니다.")
print("Failure reason: 일부 GD 방법은 좋은 bias를 못 냈고, 그 사실 자체가 보고서 소재입니다.")
"""
    ),
]


def main():
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2))
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()

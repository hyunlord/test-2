# SparseTap Experiment Design

## Overview

This project uses a two-track experimental structure:

1. Track A: Predictive Modeling
2. Track B: Rule Recovery

Both tracks are evaluated with shared metrics and feed into one final answer-selection stage.

## Track Structure

```text
Track A (Prediction)
  - Statistical scans
  - Logistic Regression (L1, GD-based optimizer)
  - NextBitMLP (GD)
  - NextBitCNN (GD)
  - Optional boosting

Track B (Rule Recovery)
  - WHT exhaustive scans
  - Prefix filtering
  - Likelihood scoring
  - Greedy / Beam / SA
  - SoftMaskGD
  - CosParityGD
  - GumbelMaskGD
  - REINFORCE
```

## Evaluation Metrics

| Metric | Meaning | Role |
|---|---|---|
| `prefix_consistency` | Exact clean-prefix validity | Hard filter |
| `log_likelihood` | Noisy-data fit under p=0.2 | Main ranking score |
| `accuracy` | Noisy next-bit accuracy | Secondary metric |
| `bias` | +/-1 correlation with parity rule | Screening signal |
| `num_taps` | Sparsity | Tie-break preference |

Shared ranking tuple:

```python
(prefix_consistency, log_likelihood, accuracy, -num_taps)
```

## GD Model Mapping

| Model | Track | Purpose |
|---|---|---|
| Logistic L1 | A + B bridge | Sparse lag prior |
| SoftMaskGD | B | Differentiable tap recovery |
| CosParityGD | B | Smooth parity approximation |
| GumbelMaskGD | B | Discrete mask learning |
| REINFORCE | B | Hard mask search with policy gradients |
| NextBitMLP | A | Predictive baseline |
| NextBitCNN | A | Predictive baseline with local context |

## Parallel Execution Plan

### Phase 1
- Load and validate data
- Build holdout split
- Build temporal validation split

### Phase 2
- Run statistical scans
- Run Logistic L1 / boosting
- Run MLP / CNN

### Phase 3
- Run WHT scans
- Run prefix + greedy + beam + SA
- Run 4 GD rule-recovery methods

### Phase 4
- Re-score all candidates
- Save comparison tables
- Select final answer

## Report Plan

### Page 1
- Problem framing
- Track A summary
- Track B summary
- Why GD models were included

### Page 2
- Candidate comparison table
- Failed attempts and lessons
- Final selected tap set and answer

## Reproducibility Checklist

- Seed fixed to 42 in NumPy / random / torch
- Shared config snapshot saved
- Candidate tables saved as CSV / JSON
- Final answer saved to text file
- Failed attempts logged to JSONL

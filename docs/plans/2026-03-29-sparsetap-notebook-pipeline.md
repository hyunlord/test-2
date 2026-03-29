# SparseTap Notebook Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a readable, independently runnable six-notebook experimental pipeline plus shared Python modules for SparseTap rule recovery and prediction.

**Architecture:** Shared logic lives in a small `src/sparsetap` package so notebooks stay focused on explanation, plots, and results. Core behavior is covered by lightweight pytest tests first, then each notebook imports the same utilities and saves intermediate artifacts for independent reruns.

**Tech Stack:** Python, NumPy, pandas, scikit-learn, xgboost/lightgbm fallback, optional PyTorch, pytest, Jupyter notebooks

---

### Task 1: Create project skeleton

**Files:**
- Create: `src/sparsetap/__init__.py`
- Create: `src/sparsetap/data.py`
- Create: `src/sparsetap/scoring.py`
- Create: `src/sparsetap/statistical.py`
- Create: `src/sparsetap/search.py`
- Create: `src/sparsetap/models.py`
- Create: `src/sparsetap/io.py`
- Create: `tests/test_core.py`
- Create: `notebooks/00_setup.ipynb`
- Create: `notebooks/01_statistical_baselines.ipynb`
- Create: `notebooks/02_rule_recovery.ipynb`
- Create: `notebooks/03_ml_baselines.ipynb`
- Create: `notebooks/04_candidate_evaluation.ipynb`
- Create: `notebooks/05_final_prediction.ipynb`
- Create: `artifacts/.gitkeep`

**Step 1: Write the failing test**

Add pytest coverage for dataset loading, lagged dataset construction, prefix consistency, likelihood, and simple tap rollout.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core.py -v`
Expected: FAIL with missing module/function errors.

**Step 3: Write minimal implementation**

Implement the shared package and exported functions required by the notebooks.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_core.py -v`
Expected: PASS

**Step 5: Commit**

Not applicable in this workspace because it is not a git repository.

### Task 2: Build readable notebook flow

**Files:**
- Create: `notebooks/00_setup.ipynb`
- Create: `notebooks/01_statistical_baselines.ipynb`
- Create: `notebooks/02_rule_recovery.ipynb`
- Create: `notebooks/03_ml_baselines.ipynb`
- Create: `notebooks/04_candidate_evaluation.ipynb`
- Create: `notebooks/05_final_prediction.ipynb`

**Step 1: Write the failing test**

Add a lightweight validation step that each notebook is valid JSON and has at least one code cell.

**Step 2: Run test to verify it fails**

Run: `python - <<'PY' ...`
Expected: FAIL before notebooks exist.

**Step 3: Write minimal implementation**

Create six notebooks with explanatory markdown and runnable code cells that import from `src/sparsetap`.

**Step 4: Run test to verify it passes**

Run: `python - <<'PY' ...`
Expected: PASS

**Step 5: Commit**

Not applicable in this workspace because it is not a git repository.

### Task 3: Verify end-to-end usability

**Files:**
- Modify: `src/sparsetap/*.py`
- Modify: `notebooks/*.ipynb`

**Step 1: Write the failing test**

Create a smoke verification script that imports the package, loads `DAY2_data.txt`, constructs a small candidate list, and checks ranking output shape.

**Step 2: Run test to verify it fails**

Run: `python scripts-or-inline-smoke-check`
Expected: FAIL until package and notebooks are wired correctly.

**Step 3: Write minimal implementation**

Adjust imports, artifact directories, and helper functions so the smoke check passes without interactive notebook state.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_core.py -v && python smoke_check`
Expected: PASS

**Step 5: Commit**

Not applicable in this workspace because it is not a git repository.

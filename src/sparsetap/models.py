import warnings

import numpy as np

from .scoring import score_candidate


def _safe_coef_order(coef):
    taps = np.where(np.abs(coef) > 1e-8)[0] + 1
    return taps.tolist()


def run_logistic_l1(dataset, data, prefix, c_grid=None, noise=0.2, max_taps=16):
    from sklearn.linear_model import LogisticRegression

    if c_grid is None:
        c_grid = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    X = dataset["X"]
    y = dataset["y"]
    candidates = []
    for c_value in c_grid:
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=c_value,
            max_iter=200,
            random_state=42,
            n_jobs=None,
            l1_ratio=1.0,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            model.fit(X, y)
        coef = model.coef_[0]
        taps = _safe_coef_order(coef)
        taps = sorted(taps, key=lambda lag: abs(coef[lag - 1]), reverse=True)[:max_taps]
        taps = sorted(taps)
        candidates.append(
            score_candidate(
                {
                    "taps": taps,
                    "method": "logistic_l1",
                    "scores": {
                        "intercept": float(model.intercept_[0]),
                        "nonzero_coefficients": int((np.abs(coef) > 1e-8).sum()),
                    },
                    "metadata": {"C": c_value},
                },
                data=data,
                prefix=prefix,
                noise=noise,
            )
        )
    return candidates


def run_xgboost(dataset, data, prefix, top_k=12, noise=0.2, max_taps=16):
    X = dataset["X"]
    y = dataset["y"]

    model = None
    backend = None
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            random_state=42,
            n_jobs=4,
            eval_metric="logloss",
        )
        backend = "xgboost"
    except ImportError:
        try:
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=4,
                verbose=-1,
            )
            backend = "lightgbm"
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            model = HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_depth=4,
                max_iter=120,
                random_state=42,
            )
            backend = "sklearn_histgb"

    model.fit(X, y)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.ones(X.shape[1], dtype=np.float32)
    ranking = np.argsort(importances)[::-1] + 1

    candidates = []
    for size in range(1, min(max_taps, top_k) + 1):
        taps = sorted(ranking[:size].tolist())
        candidates.append(
            score_candidate(
                {
                    "taps": taps,
                    "method": "xgboost",
                    "scores": {"feature_importance_sum": float(importances[ranking[:size] - 1].sum())},
                    "metadata": {"backend": backend, "size": size},
                },
                data=data,
                prefix=prefix,
                noise=noise,
            )
        )
    return candidates


def run_mlp(dataset, data, prefix, hidden_dim=64, epochs=15, noise=0.2, max_taps=16):
    try:
        import torch
        from torch import nn
    except ImportError:
        return []

    X = torch.tensor(dataset["X"], dtype=torch.float32)
    y = torch.tensor(dataset["y"].astype(np.float32)).unsqueeze(1)

    model = nn.Sequential(
        nn.Linear(X.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        first_layer = model[0].weight.abs().mean(dim=0).cpu().numpy()
    ranking = np.argsort(first_layer)[::-1] + 1
    candidates = []
    for size in [2, 4, 8, min(max_taps, 12)]:
        size = min(size, max_taps, len(ranking))
        taps = sorted(ranking[:size].tolist())
        candidates.append(
            score_candidate(
                {
                    "taps": taps,
                    "method": "mlp",
                    "scores": {"importance_mass": float(first_layer[ranking[:size] - 1].sum())},
                    "metadata": {"hidden_dim": hidden_dim, "epochs": epochs, "size": size},
                },
                data=data,
                prefix=prefix,
                noise=noise,
            )
        )
    return candidates

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


@dataclass
class GDResult:
    method: str = ""
    offsets: list = field(default_factory=list)
    mask_values: Optional[np.ndarray] = None
    loss_history: list = field(default_factory=list)
    bias: float = 0.0
    accuracy: float = 0.0
    elapsed_seconds: float = 0.0
    converged: bool = False
    metadata: dict = field(default_factory=dict)


def _fix_torch_seed(seed: int):
    if not HAS_TORCH:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_pm_windows(data_pm: np.ndarray, max_w: int):
    data_pm = np.asarray(data_pm, dtype=np.float32)
    seq_len = data_pm.shape[1]
    rows = data_pm.shape[0] * (seq_len - max_w)
    X = np.empty((rows, max_w), dtype=np.float32)
    for lag in range(1, max_w + 1):
        X[:, lag - 1] = data_pm[:, max_w - lag : seq_len - lag].reshape(-1)
    y = data_pm[:, max_w:].reshape(-1).astype(np.float32)
    return X, y


def _build_01_windows(data_01: np.ndarray, max_w: int):
    data_01 = np.asarray(data_01, dtype=np.float32)
    seq_len = data_01.shape[1]
    rows = data_01.shape[0] * (seq_len - max_w)
    X = np.empty((rows, max_w), dtype=np.float32)
    for lag in range(1, max_w + 1):
        X[:, lag - 1] = data_01[:, max_w - lag : seq_len - lag].reshape(-1)
    y = data_01[:, max_w:].reshape(-1).astype(np.float32)
    return X, y


def _eval_bias(offsets, data_pm, max_w):
    if not offsets:
        return 0.0
    X_pm, y_pm = _build_pm_windows(data_pm, max_w)
    selected = np.ones(X_pm.shape[0], dtype=np.float32)
    for offset in offsets:
        selected *= X_pm[:, offset - 1]
    return float(np.mean(selected * y_pm))


def _eval_accuracy_from_offsets(offsets, data_01, max_w):
    if not offsets:
        return 0.0
    X_01, y_01 = _build_01_windows(data_01, max_w)
    parity = np.zeros(X_01.shape[0], dtype=np.uint8)
    for offset in offsets:
        parity ^= X_01[:, offset - 1].astype(np.uint8)
    return float(np.mean(parity == y_01.astype(np.uint8)))


def _finalize_result(method, offsets, mask_values, loss_history, data_01, data_pm, max_w, elapsed, metadata=None):
    offsets = sorted(set(int(x) for x in offsets))
    bias = _eval_bias(offsets, data_pm, max_w=max_w)
    accuracy = _eval_accuracy_from_offsets(offsets, data_01, max_w=max_w)
    return GDResult(
        method=method,
        offsets=offsets,
        mask_values=None if mask_values is None else np.asarray(mask_values, dtype=np.float32),
        loss_history=[float(x) for x in loss_history],
        bias=float(bias),
        accuracy=float(accuracy),
        elapsed_seconds=float(elapsed),
        converged=bool(loss_history),
        metadata=metadata or {},
    )


def _candidate_offsets_from_mask(mask_values: np.ndarray, max_offsets: int = 16, threshold: float = 0.5):
    mask_values = np.asarray(mask_values, dtype=np.float32)
    offsets = [idx + 1 for idx, value in enumerate(mask_values) if value >= threshold]
    if not offsets:
        offsets = (np.argsort(mask_values)[::-1][:max_offsets] + 1).tolist()
    return sorted(offsets[:max_offsets])


def run_soft_mask_gd(data_pm, max_w=64, n_epochs=500, batch_size=128, lr=0.1, l1_lambda=0.005, seed=42) -> GDResult:
    if not HAS_TORCH:
        return GDResult(method="soft_mask_gd", metadata={"error": "no torch"})

    start = time.perf_counter()
    _fix_torch_seed(seed)
    data_pm = np.asarray(data_pm, dtype=np.float32)
    data_01 = ((1.0 - data_pm) / 2.0).astype(np.float32)
    X_np, y_np = _build_pm_windows(data_pm, max_w=max_w)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    logits = torch.nn.Parameter(torch.zeros(max_w))
    optimizer = torch.optim.Adam([logits], lr=lr)
    loss_history = []
    noise_corr = 1.0 - 2.0 * 0.2

    for _ in range(n_epochs):
        perm = torch.randperm(X.shape[0])
        epoch_loss = 0.0
        for batch_idx in perm.split(batch_size):
            xb = X[batch_idx]
            yb = y[batch_idx]
            mask = torch.sigmoid(logits)
            pred = torch.prod((1.0 - mask) + mask * xb, dim=1)
            p = 0.5 * (1.0 + yb * noise_corr * pred)
            p = torch.clamp(p, min=1e-6, max=1.0)
            loss = -torch.log(p).mean() + l1_lambda * mask.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        loss_history.append(epoch_loss)

    mask_values = torch.sigmoid(logits).detach().cpu().numpy()
    offsets = _candidate_offsets_from_mask(mask_values)
    return _finalize_result(
        "soft_mask_gd",
        offsets,
        mask_values,
        loss_history,
        data_01,
        data_pm,
        max_w,
        time.perf_counter() - start,
        metadata={"seed": seed, "lr": lr, "l1_lambda": l1_lambda},
    )


def run_cos_parity_gd(data_01, data_pm, max_w=64, n_steps=3000, batch_size=10000, lr=0.05, l1_lambda=0.001, seed=42) -> GDResult:
    if not HAS_TORCH:
        return GDResult(method="cos_parity_gd", metadata={"error": "no torch"})

    start = time.perf_counter()
    _fix_torch_seed(seed)
    data_01 = np.asarray(data_01, dtype=np.float32)
    data_pm = np.asarray(data_pm, dtype=np.float32)
    X_np, y_np = _build_01_windows(data_01, max_w=max_w)
    X = torch.tensor(X_np, dtype=torch.float32)
    y_pm = torch.tensor(_build_pm_windows(data_pm, max_w=max_w)[1], dtype=torch.float32)
    coeff = torch.nn.Parameter(torch.zeros(max_w))
    optimizer = torch.optim.Adam([coeff], lr=lr)
    loss_history = []

    for step in range(n_steps):
        idx = torch.randint(0, X.shape[0], (min(batch_size, X.shape[0]),))
        xb = X[idx]
        yb = y_pm[idx]
        mask = torch.sigmoid(coeff * 5.0)
        pred = torch.cos(torch.pi * torch.sum(mask * xb, dim=1))
        bias = torch.mean(yb * pred)
        loss = -bias + l1_lambda * mask.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, n_steps // 100) == 0:
            loss_history.append(float(loss.item()))

    mask_values = torch.sigmoid(coeff * 5.0).detach().cpu().numpy()
    offsets = _candidate_offsets_from_mask(mask_values)
    return _finalize_result(
        "cos_parity_gd",
        offsets,
        mask_values,
        loss_history,
        data_01,
        data_pm,
        max_w,
        time.perf_counter() - start,
        metadata={"seed": seed, "lr": lr, "l1_lambda": l1_lambda},
    )


def run_gumbel_mask_gd(
    data_pm,
    max_w=64,
    n_steps=3000,
    batch_size=256,
    lr=0.1,
    l1_lambda=0.0005,
    temp_start=5.0,
    temp_end=0.1,
    seed=42,
) -> GDResult:
    if not HAS_TORCH:
        return GDResult(method="gumbel_mask_gd", metadata={"error": "no torch"})

    start = time.perf_counter()
    _fix_torch_seed(seed)
    data_pm = np.asarray(data_pm, dtype=np.float32)
    data_01 = ((1.0 - data_pm) / 2.0).astype(np.float32)
    X_np, y_np = _build_pm_windows(data_pm, max_w=max_w)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    logits = torch.nn.Parameter(torch.zeros(max_w))
    optimizer = torch.optim.Adam([logits], lr=lr)
    loss_history = []

    for step in range(n_steps):
        temp = temp_start * (temp_end / temp_start) ** (step / max(1, n_steps - 1))
        idx = torch.randint(0, X.shape[0], (min(batch_size, X.shape[0]),))
        xb = X[idx]
        yb = y[idx]
        noise = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(torch.clamp(noise, 1e-6, 1 - 1e-6)))
        soft_mask = torch.sigmoid((logits + gumbel) / temp)
        if step >= int(0.7 * n_steps):
            hard = (soft_mask > 0.5).float()
            mask = hard.detach() - soft_mask.detach() + soft_mask
        else:
            mask = soft_mask
        pred = torch.prod((1.0 - mask) + mask * xb, dim=1)
        bias = torch.mean(yb * pred)
        loss = -(bias**2) + l1_lambda * torch.sum(torch.sigmoid(logits))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, n_steps // 100) == 0:
            loss_history.append(float(loss.item()))

    mask_values = torch.sigmoid(logits).detach().cpu().numpy()
    offsets = _candidate_offsets_from_mask(mask_values)
    return _finalize_result(
        "gumbel_mask_gd",
        offsets,
        mask_values,
        loss_history,
        data_01,
        data_pm,
        max_w,
        time.perf_counter() - start,
        metadata={"seed": seed, "lr": lr, "temp_start": temp_start, "temp_end": temp_end},
    )


def run_reinforce(data_pm, max_w=64, n_steps=5000, batch_size=128, n_samples=32, lr=0.05, l1_lambda=0.01, seed=42) -> GDResult:
    if not HAS_TORCH:
        return GDResult(method="reinforce", metadata={"error": "no torch"})

    start = time.perf_counter()
    _fix_torch_seed(seed)
    data_pm = np.asarray(data_pm, dtype=np.float32)
    data_01 = ((1.0 - data_pm) / 2.0).astype(np.float32)
    X_np, y_np = _build_pm_windows(data_pm, max_w=max_w)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    logits = torch.nn.Parameter(torch.zeros(max_w))
    optimizer = torch.optim.Adam([logits], lr=lr)
    baseline = 0.0
    loss_history = []

    for step in range(n_steps):
        idx = torch.randint(0, X.shape[0], (min(batch_size, X.shape[0]),))
        xb = X[idx]
        yb = y[idx]
        probs = torch.sigmoid(logits).unsqueeze(0).expand(n_samples, -1)
        bernoulli = torch.distributions.Bernoulli(probs=probs)
        samples = bernoulli.sample()
        selected = torch.prod((1.0 - samples[:, None, :]) + samples[:, None, :] * xb[None, :, :], dim=2)
        rewards = torch.abs(torch.mean(yb[None, :] * selected, dim=1))
        baseline = 0.9 * baseline + 0.1 * float(rewards.mean().item())
        advantage = rewards - baseline
        log_prob = bernoulli.log_prob(samples).sum(dim=1)
        loss = -(advantage.detach() * log_prob).mean() + l1_lambda * torch.sigmoid(logits).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, n_steps // 100) == 0:
            loss_history.append(float(loss.item()))

    mask_values = torch.sigmoid(logits).detach().cpu().numpy()
    offsets = _candidate_offsets_from_mask(mask_values)
    return _finalize_result(
        "reinforce",
        offsets,
        mask_values,
        loss_history,
        data_01,
        data_pm,
        max_w,
        time.perf_counter() - start,
        metadata={"seed": seed, "lr": lr, "n_samples": n_samples},
    )


def run_next_bit_mlp(data_01, max_w=64, hidden_dim=128, n_epochs=30, batch_size=2048, lr=0.001, seed=42) -> GDResult:
    if not HAS_TORCH:
        return GDResult(method="next_bit_mlp", metadata={"error": "no torch"})

    start = time.perf_counter()
    _fix_torch_seed(seed)
    data_01 = np.asarray(data_01, dtype=np.float32)
    data_pm = 1.0 - 2.0 * data_01
    X_np, y_np = _build_01_windows(data_01, max_w=max_w)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

    model = nn.Sequential(
        nn.Linear(max_w, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_history = []

    for _ in range(n_epochs):
        perm = torch.randperm(X.shape[0])
        epoch_loss = 0.0
        for batch_idx in perm.split(batch_size):
            xb = X[batch_idx]
            yb = y[batch_idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        loss_history.append(epoch_loss)

    with torch.no_grad():
        importance = model[0].weight.abs().mean(dim=0).cpu().numpy()
        preds = (torch.sigmoid(model(X)).squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
    offsets = (np.argsort(importance)[::-1][:16] + 1).tolist()
    result = _finalize_result(
        "next_bit_mlp",
        offsets,
        importance,
        loss_history,
        data_01,
        data_pm,
        max_w,
        time.perf_counter() - start,
        metadata={"seed": seed, "hidden_dim": hidden_dim, "lr": lr},
    )
    result.accuracy = float(np.mean(preds == y_np.astype(np.uint8)))
    return result


def run_next_bit_cnn(data_01, max_w=64, n_epochs=30, batch_size=2048, lr=0.001, seed=42) -> GDResult:
    if not HAS_TORCH:
        return GDResult(method="next_bit_cnn", metadata={"error": "no torch"})

    start = time.perf_counter()
    _fix_torch_seed(seed)
    data_01 = np.asarray(data_01, dtype=np.float32)
    data_pm = 1.0 - 2.0 * data_01
    X_np, y_np = _build_01_windows(data_01, max_w=max_w)
    X = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

    class NextBitCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
            self.fc = nn.Linear(32 * max_w, 1)

        def forward(self, xb):
            x = torch.relu(self.conv1(xb))
            x = torch.relu(self.conv2(x))
            return self.fc(x.flatten(1))

    model = NextBitCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_history = []

    for _ in range(n_epochs):
        perm = torch.randperm(X.shape[0])
        epoch_loss = 0.0
        for batch_idx in perm.split(batch_size):
            xb = X[batch_idx]
            yb = y[batch_idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        loss_history.append(epoch_loss)

    X_grad = torch.tensor(X_np, dtype=torch.float32, requires_grad=True).unsqueeze(1)
    logits = model(X_grad)
    probs = torch.sigmoid(logits).mean()
    probs.backward()
    importance = X_grad.grad.abs().mean(dim=0).squeeze(0).mean(dim=0).detach().cpu().numpy()

    with torch.no_grad():
        preds = (torch.sigmoid(model(X)).squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
    offsets = (np.argsort(importance)[::-1][:16] + 1).tolist()
    result = _finalize_result(
        "next_bit_cnn",
        offsets,
        importance,
        loss_history,
        data_01,
        data_pm,
        max_w,
        time.perf_counter() - start,
        metadata={"seed": seed, "lr": lr},
    )
    result.accuracy = float(np.mean(preds == y_np.astype(np.uint8)))
    return result

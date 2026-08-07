"""
Differential Privacy — DP-SGD with Moments Accountant.
Pure Python: gradient clipping, Gaussian noise, epsilon tracking.
"""
import math
import random

from app.ml.logistic_regression import (
    compute_gradients, sgd_step, predict_proba, binary_cross_entropy,
    compute_accuracy, init_weights,
)
from app.ml.fl_algorithms import weighted_average, scalar_avg


# ── Gradient Clipping ────────────────────────────────────────────────────────────[...]

def clip_gradient(grad_w: list[float], grad_b: float, clip_norm: float) -> tuple[list[float], float]:
    """Per-sample gradient clipping to bound L2 sensitivity."""
    norm = math.sqrt(sum(g ** 2 for g in grad_w) + grad_b ** 2)
    if norm > clip_norm:
        scale = clip_norm / norm
        grad_w = [g * scale for g in grad_w]
        grad_b = grad_b * scale
    return grad_w, grad_b


# ── Gaussian Noise ─────────────────────────────────────────────────────────────[...]

def add_gaussian_noise(
    grad_w: list[float],
    grad_b: float,
    clip_norm: float,
    noise_multiplier: float,
    n: int,
) -> tuple[list[float], float]:
    """
    Add calibrated Gaussian noise.
    σ = noise_multiplier * clip_norm / n
    """
    sigma = noise_multiplier * clip_norm / n
    noisy_w = [g + random.gauss(0, sigma) for g in grad_w]
    noisy_b = grad_b + random.gauss(0, sigma)
    return noisy_w, noisy_b


# ── Moments Accountant (simplified Rényi DP → (ε, δ)-DP) ─────────────────────────

def compute_epsilon(
    noise_multiplier: float,
    n_samples: int,
    batch_size: int,
    n_steps: int,
    delta: float = 1e-5,
) -> float:
    """
    Simplified moments accountant approximation.
    Uses the Gaussian mechanism with subsampling.
    Reference: Abadi et al. 2016.
    """
    if noise_multiplier <= 0:
        return float("inf")
    q = batch_size / n_samples  # sampling ratio
    # Rényi divergence at order alpha
    best_eps = float("inf")
    for alpha in range(2, 200):
        # log moment: alpha * q^2 / (2 * sigma^2) * T (simplified)
        log_moment = alpha * (q ** 2) * n_steps / (2.0 * noise_multiplier ** 2)
        eps_rdp = log_moment + math.log(1 / delta) / (alpha - 1)
        if eps_rdp < best_eps:
            best_eps = eps_rdp
    return round(best_eps, 4)


# ── DP-SGD Local Training ─────────────────────────────────────────────────────────

def dpsgd_local_train(
    weights: list[float],
    bias: float,
    X: list[list[float]],
    y: list[int],
    lr: float = 0.01,
    epochs: int = 5,
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    batch_size: int = 32,
) -> tuple[list[float], float, float]:
    """
    DP-SGD local training:
    1. Compute per-sample gradients
    2. Clip each gradient by clip_norm
    3. Add Gaussian noise
    4. Average and update
    """
    n = len(X)
    for _ in range(epochs):
        # Mini-batch
        indices = random.sample(range(n), min(batch_size, n))
        X_batch = [X[i] for i in indices]
        y_batch = [y[i] for i in indices]

        # Per-sample gradients + clipping
        clipped_grads_w = []
        clipped_grads_b = []
        for xi, yi in zip(X_batch, y_batch):
            gw, gb = compute_gradients(weights, bias, [xi], [yi])
            gw_c, gb_c = clip_gradient(gw, gb, clip_norm)
            clipped_grads_w.append(gw_c)
            clipped_grads_b.append(gb_c)

        # Sum clipped gradients
        n_batch = len(X_batch)
        sum_gw = [sum(clipped_grads_w[i][j] for i in range(n_batch)) for j in range(len(weights))]
        sum_gb = sum(clipped_grads_b)

        # Add Gaussian noise + normalize
        noisy_gw, noisy_gb = add_gaussian_noise(sum_gw, sum_gb, clip_norm, noise_multiplier, n_batch)
        avg_gw = [g / n_batch for g in noisy_gw]
        avg_gb = noisy_gb / n_batch

        weights, bias = sgd_step(weights, bias, avg_gw, avg_gb, lr)

    probs = predict_proba(weights, bias, X)
    loss = binary_cross_entropy(y, probs)
    return weights, bias, loss


# ── DP-SGD Federated Training ─────────────────────────────────────────────────────

def run_dpsgd(
    clients: list[tuple],
    n_rounds: int = 20,
    lr: float = 0.01,
    local_epochs: int = 5,
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    delta: float = 1e-5,
    batch_size: int = 32,
    X_val: list = None,
    y_val: list = None,
) -> tuple[list[dict], list[float], float, list[dict]]:
    """
    Federated DP-SGD:
    Each client trains with DP-SGD; server aggregates via FedAvg.
    Tracks cumulative epsilon budget per round.
    """
    n_features = len(clients[0][0][0])
    global_w, global_b = init_weights(n_features)
    history = []
    privacy_history = []
    cumulative_steps = 0

    for r in range(n_rounds):
        client_ws, client_bs, client_sizes = [], [], []
        for X_k, y_k in clients:
            if not X_k:
                continue
            w_k, b_k, _ = dpsgd_local_train(
                list(global_w), global_b, X_k, y_k,
                lr, local_epochs, clip_norm, noise_multiplier, batch_size,
            )
            client_ws.append(w_k)
            client_bs.append(b_k)
            client_sizes.append(len(X_k))

        global_w = weighted_average(client_ws, client_sizes)
        global_b = scalar_avg(client_bs, client_sizes)
        cumulative_steps += local_epochs

        # Privacy accounting
        avg_n = int(sum(client_sizes) / len(client_sizes))
        eps = compute_epsilon(noise_multiplier, avg_n, batch_size, cumulative_steps, delta)
        privacy_history.append({
            "round": r + 1,
            "epsilon": eps,
            "delta": delta,
            "noise_multiplier": noise_multiplier,
            "clip_norm": clip_norm,
        })

        all_X = [x for X_k, _ in clients for x in X_k]
        all_y = [yi for _, y_k in clients for yi in y_k]
        probs = predict_proba(global_w, global_b, all_X)
        loss = binary_cross_entropy(all_y, probs)
        preds = [1 if p >= 0.5 else 0 for p in probs]
        acc = compute_accuracy(all_y, preds)
        round_data = {
            "round": r + 1,
            "loss": round(loss, 4),
            "accuracy": round(acc, 4),
            "num_clients": len(client_sizes),
            "epsilon": eps,
        }
        if X_val and y_val:
            probs_v = predict_proba(global_w, global_b, X_val)
            val_loss = binary_cross_entropy(y_val, probs_v)
            val_preds = [1 if p >= 0.5 else 0 for p in probs_v]
            round_data["val_loss"] = round(val_loss, 4)
            round_data["val_accuracy"] = round(compute_accuracy(y_val, val_preds), 4)
        history.append(round_data)

    return history, global_w, global_b, privacy_history

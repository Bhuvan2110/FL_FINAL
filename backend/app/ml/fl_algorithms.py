"""
Federated Learning algorithms — pure Python, zero ML libraries.
Implements: FedAvg, FedProx, SCAFFOLD, Central Training.
"""
from app.ml.logistic_regression import (
    train_local, compute_gradients, sgd_step, predict_proba,
    binary_cross_entropy, compute_accuracy, predict, init_weights,
)


# ── Helpers ───────────────────────────────────────────────────────────────[...]

def weighted_average(weight_lists: list[list[float]], counts: list[int]) -> list[float]:
    """FedAvg weighted average: sum(n_k * w_k) / sum(n_k)."""
    total = sum(counts)
    n_params = len(weight_lists[0])
    avg = [0.0] * n_params
    for weights, n in zip(weight_lists, counts):
        for j in range(n_params):
            avg[j] += n * weights[j] / total
    return avg


def scalar_avg(bias_list: list[float], counts: list[int]) -> float:
    total = sum(counts)
    return sum(b * n for b, n in zip(bias_list, counts)) / total


# ── Central Training ────────────────────────────────────────────────────────────[...]

def run_central(
    X: list[list[float]],
    y: list[int],
    n_rounds: int = 20,
    lr: float = 0.05,
    epochs_per_round: int = 5,
    X_val: list = None,
    y_val: list = None,
) -> list[dict]:
    """Train on all data centrally — baseline comparison."""
    n_features = len(X[0])
    weights, bias = init_weights(n_features)
    history = []
    for r in range(n_rounds):
        weights, bias, loss = train_local(weights, bias, X, y, lr, epochs_per_round)
        preds = predict(weights, bias, X)
        acc = compute_accuracy(y, preds)
        round_data = {"round": r + 1, "loss": round(loss, 4), "accuracy": round(acc, 4)}
        if X_val and y_val:
            probs_v = predict_proba(weights, bias, X_val)
            val_loss = binary_cross_entropy(y_val, probs_v)
            val_preds = [1 if p >= 0.5 else 0 for p in probs_v]
            round_data["val_loss"] = round(val_loss, 4)
            round_data["val_accuracy"] = round(compute_accuracy(y_val, val_preds), 4)
        history.append(round_data)
    return history, weights, bias


# ── FedAvg ────────────────────────────────────────────────────────────[...]

def run_fedavg(
    clients: list[tuple],          # [(X_k, y_k), ...]
    n_rounds: int = 20,
    lr: float = 0.01,
    local_epochs: int = 5,
    X_val: list = None,
    y_val: list = None,
) -> tuple[list[dict], list[float], float]:
    """
    FedAvg: McMahan et al. 2017.
    Server broadcasts global weights; clients train locally;
    server aggregates via weighted average.
    """
    n_features = len(clients[0][0][0])
    global_w, global_b = init_weights(n_features)
    history = []

    for r in range(n_rounds):
        client_ws, client_bs, client_sizes = [], [], []
        for X_k, y_k in clients:
            if not X_k:
                continue
            w_k, b_k, _ = train_local(list(global_w), global_b, X_k, y_k, lr, local_epochs)
            client_ws.append(w_k)
            client_bs.append(b_k)
            client_sizes.append(len(X_k))

        global_w = weighted_average(client_ws, client_sizes)
        global_b = scalar_avg(client_bs, client_sizes)

        # Compute global metrics
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
        }
        if X_val and y_val:
            probs_v = predict_proba(global_w, global_b, X_val)
            val_loss = binary_cross_entropy(y_val, probs_v)
            val_preds = [1 if p >= 0.5 else 0 for p in probs_v]
            round_data["val_loss"] = round(val_loss, 4)
            round_data["val_accuracy"] = round(compute_accuracy(y_val, val_preds), 4)
        history.append(round_data)

    return history, global_w, global_b


# ── FedProx ───────────────────────────────────────────────────────────────[...]

def fedprox_local_train(
    weights: list[float],
    bias: float,
    global_w: list[float],
    global_b: float,
    X: list[list[float]],
    y: list[int],
    lr: float = 0.01,
    mu: float = 0.1,
    epochs: int = 5,
) -> tuple[list[float], float, float]:
    """
    FedProx local update with proximal term:
    min F_k(w) + (mu/2) * ||w - w_global||²
    """
    for _ in range(epochs):
        grad_w, grad_b = compute_gradients(weights, bias, X, y)
        # Add proximal gradient
        prox_grad = [gw + mu * (w - gw_) for gw, w, gw_ in zip(grad_w, weights, global_w)]
        prox_grad_b = grad_b + mu * (bias - global_b)
        weights, bias = sgd_step(weights, bias, prox_grad, prox_grad_b, lr)
    probs = predict_proba(weights, bias, X)
    loss = binary_cross_entropy(y, probs)
    return weights, bias, loss


def run_fedprox(
    clients: list[tuple],
    n_rounds: int = 20,
    lr: float = 0.01,
    mu: float = 0.1,
    local_epochs: int = 5,
    X_val: list = None,
    y_val: list = None,
) -> tuple[list[dict], list[float], float]:
    """
    FedProx: Li et al. 2020.
    Like FedAvg but with proximal regularization for non-IID stability.
    """
    n_features = len(clients[0][0][0])
    global_w, global_b = init_weights(n_features)
    history = []

    for r in range(n_rounds):
        client_ws, client_bs, client_sizes = [], [], []
        for X_k, y_k in clients:
            if not X_k:
                continue
            w_k, b_k, _ = fedprox_local_train(
                list(global_w), global_b, global_w, global_b, X_k, y_k, lr, mu, local_epochs
            )
            client_ws.append(w_k)
            client_bs.append(b_k)
            client_sizes.append(len(X_k))

        global_w = weighted_average(client_ws, client_sizes)
        global_b = scalar_avg(client_bs, client_sizes)

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
            "mu": mu,
        }
        if X_val and y_val:
            probs_v = predict_proba(global_w, global_b, X_val)
            val_loss = binary_cross_entropy(y_val, probs_v)
            val_preds = [1 if p >= 0.5 else 0 for p in probs_v]
            round_data["val_loss"] = round(val_loss, 4)
            round_data["val_accuracy"] = round(compute_accuracy(y_val, val_preds), 4)
        history.append(round_data)

    return history, global_w, global_b


# ── SCAFFOLD ───────────────────────────────────────────────────────────────[...]

def run_scaffold(
    clients: list[tuple],
    n_rounds: int = 20,
    lr: float = 0.01,
    local_epochs: int = 5,
    X_val: list = None,
    y_val: list = None,
) -> tuple[list[dict], list[float], float]:
    """
    SCAFFOLD: Karimireddy et al. 2020.
    Server + client control variates correct for client drift in non-IID settings.
    c_i: client control variate
    c:   server control variate
    Local update: w_k ← w_k - lr * (grad + c - c_i)
    """
    n_features = len(clients[0][0][0])
    global_w, global_b = init_weights(n_features)

    # Control variates
    server_cv = [0.0] * n_features
    server_cv_b = 0.0
    client_cvs = [[0.0] * n_features for _ in clients]
    client_cvs_b = [0.0] * len(clients)

    history = []

    for r in range(n_rounds):
        client_ws, client_bs, client_sizes = [], [], []
        new_client_cvs, new_client_cvs_b = [], []

        for k, (X_k, y_k) in enumerate(clients):
            if not X_k:
                continue
            w_k = list(global_w)
            b_k = global_b
            c_k = client_cvs[k]
            c_k_b = client_cvs_b[k]

            for _ in range(local_epochs):
                grad_w, grad_b = compute_gradients(w_k, b_k, X_k, y_k)
                # Corrected gradient: grad + server_cv - client_cv
                corrected_grad = [gw + sc - ck for gw, sc, ck in zip(grad_w, server_cv, c_k)]
                corrected_grad_b = grad_b + server_cv_b - c_k_b
                w_k, b_k = sgd_step(w_k, b_k, corrected_grad, corrected_grad_b, lr)

            # Update client control variate
            new_c_k = [c_k[j] - server_cv[j] + (global_w[j] - w_k[j]) / (local_epochs * lr)
                       for j in range(n_features)]
            new_c_k_b = c_k_b - server_cv_b + (global_b - b_k) / (local_epochs * lr)
            new_client_cvs.append(new_c_k)
            new_client_cvs_b.append(new_c_k_b)
            client_ws.append(w_k)
            client_bs.append(b_k)
            client_sizes.append(len(X_k))

        # Update global model
        global_w = weighted_average(client_ws, client_sizes)
        global_b = scalar_avg(client_bs, client_sizes)

        # Update server control variate: c = avg(c_k_new)
        server_cv = [sum(cv[j] for cv in new_client_cvs) / len(new_client_cvs) for j in range(n_features)]
        server_cv_b = sum(new_client_cvs_b) / len(new_client_cvs_b)
        for k, (nc, nb) in enumerate(zip(new_client_cvs, new_client_cvs_b)):
            client_cvs[k] = nc
            client_cvs_b[k] = nb

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
        }
        if X_val and y_val:
            probs_v = predict_proba(global_w, global_b, X_val)
            val_loss = binary_cross_entropy(y_val, probs_v)
            val_preds = [1 if p >= 0.5 else 0 for p in probs_v]
            round_data["val_loss"] = round(val_loss, 4)
            round_data["val_accuracy"] = round(compute_accuracy(y_val, val_preds), 4)
        history.append(round_data)

    return history, global_w, global_b

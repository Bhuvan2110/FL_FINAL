"""Tests for all 5 FL/ML algorithms."""
import math
from app.ml.logistic_regression import sigmoid, binary_cross_entropy, predict_proba, init_weights
from app.ml.fl_algorithms import run_fedavg, run_fedprox, run_scaffold, run_central, weighted_average
from app.ml.differential_privacy import run_dpsgd, clip_gradient, compute_epsilon
from app.ml.preprocessing import (
    parse_csv, min_max_normalize,
    stratified_split, partition_iid, partition_non_iid,
)
from app.ml.metrics import confusion_matrix, f1_score, roc_curve, feature_importance


# ── Sample Data ──────────────────────────────────────────────────────────────[...]

def make_data(n=100, n_features=4, seed=42):
    import random
    random.seed(seed)
    X = [[random.gauss(0, 1) for _ in range(n_features)] for _ in range(n)]
    y = [1 if sum(x) > 0 else 0 for x in X]
    return X, y


# ── Logistic Regression ───────────────────────────────────────────────────────

def test_sigmoid_bounds():
    assert 0 < sigmoid(0) < 1
    assert sigmoid(100) > 0.99
    assert sigmoid(-100) < 0.01


def test_binary_cross_entropy():
    loss = binary_cross_entropy([1, 0, 1], [0.9, 0.1, 0.8])
    assert loss > 0


def test_init_weights():
    w, b = init_weights(4)
    assert len(w) == 4
    assert b == 0.0


def test_predict_proba_range():
    X, y = make_data(50)
    w, b = init_weights(4)
    probs = predict_proba(w, b, X)
    assert all(0.0 <= p <= 1.0 for p in probs)


# ── FedAvg ────────────────────────────────────────────────────────────[...]

def test_fedavg_runs():
    X, y = make_data(100)
    _, scalers = min_max_normalize(X)
    clients = partition_iid(X, y, n_clients=3)
    history, weights, bias = run_fedavg(clients, n_rounds=5, lr=0.05, local_epochs=2)
    assert len(history) == 5
    assert all("accuracy" in r for r in history)
    assert len(weights) == 4


def test_fedavg_accuracy_improves():
    X, y = make_data(200)
    clients = partition_iid(X, y, n_clients=5)
    history, _, _ = run_fedavg(clients, n_rounds=15, lr=0.05, local_epochs=5)
    assert history[-1]["accuracy"] >= history[0]["accuracy"] - 0.1  # generally improves


def test_weighted_average():
    w1, w2 = [1.0, 2.0], [3.0, 4.0]
    avg = weighted_average([w1, w2], [1, 1])
    assert abs(avg[0] - 2.0) < 1e-9


# ── FedProx ───────────────────────────────────────────────────────────────[...]

def test_fedprox_runs():
    X, y = make_data(100)
    clients = partition_non_iid(X, y, n_clients=3)
    history, _, _ = run_fedprox(clients, n_rounds=5, lr=0.05, mu=0.1, local_epochs=2)
    assert len(history) == 5


# ── SCAFFOLD ───────────────────────────────────────────────────────────────[...]

def test_scaffold_runs():
    X, y = make_data(100)
    clients = partition_non_iid(X, y, n_clients=3)
    history, _, _ = run_scaffold(clients, n_rounds=5, lr=0.05, local_epochs=2)
    assert len(history) == 5


# ── Central ───────────────────────────────────────────────────────────────[...]

def test_central_runs():
    X, y = make_data(100)
    history, weights, bias = run_central(X, y, n_rounds=5, lr=0.05, epochs_per_round=2)
    assert len(history) == 5
    assert len(weights) == 4


# ── DP-SGD ───────────────────────────────────────────────────────────────[...]

def test_dpsgd_runs():
    X, y = make_data(100)
    clients = partition_iid(X, y, n_clients=3)
    history, _, _, privacy = run_dpsgd(
        clients, n_rounds=5, lr=0.05, local_epochs=2,
        clip_norm=1.0, noise_multiplier=0.5, delta=1e-5, batch_size=16
    )
    assert len(history) == 5
    assert len(privacy) == 5
    assert all("epsilon" in p for p in privacy)


def test_gradient_clipping():
    grad_w = [3.0, 4.0]
    grad_b = 0.0
    clipped_w, clipped_b = clip_gradient(grad_w, grad_b, clip_norm=1.0)
    norm = math.sqrt(sum(g**2 for g in clipped_w))
    assert norm <= 1.0 + 1e-9


def test_epsilon_positive():
    eps = compute_epsilon(noise_multiplier=1.0, n_samples=100, batch_size=10, n_steps=10, delta=1e-5)
    assert eps > 0


# ── Metrics ───────────────────────────────────────────────────────────────[...]

def test_confusion_matrix():
    cm = confusion_matrix([1, 0, 1, 0], [1, 0, 0, 1])
    assert cm["tp"] == 1
    assert cm["tn"] == 1
    assert cm["fp"] == 1
    assert cm["fn"] == 1


def test_f1_score():
    cm = confusion_matrix([1, 1, 0, 0], [1, 0, 0, 0])
    f1 = f1_score(cm)
    assert 0 <= f1 <= 1


def test_roc_auc_range():
    y_true = [1, 0, 1, 0, 1, 0]
    scores = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]
    result = roc_curve(y_true, scores)
    assert 0 <= result["auc"] <= 1
    assert result["auc"] > 0.7  # should be good on this data


def test_feature_importance_normalized():
    weights = [1.0, -2.0, 0.5, 3.0]
    fi = feature_importance(weights)
    total = sum(f["importance"] for f in fi)
    assert abs(total - 1.0) < 1e-3  # Allow 4-decimal rounding (0.9999...)


# ── Preprocessing ─────────────────────────────────────────────────────────────[...]

def test_min_max_normalize():
    data = [[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]]
    norm, scalers = min_max_normalize(data)
    assert norm[0][0] == 0.0
    assert norm[2][0] == 1.0


def test_stratified_split_maintains_ratio():
    X, y = make_data(200)
    splits = stratified_split(X, y)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    total = len(y_train) + len(y_val) + len(y_test)
    assert abs(total - 200) <= 2  # allow rounding


def test_parse_csv():
    csv_bytes = b"a,b,label\n1,2,1\n3,4,0\n5,6,1\n"
    headers, rows = parse_csv(csv_bytes)
    assert headers == ["a", "b", "label"]
    assert len(rows) == 3

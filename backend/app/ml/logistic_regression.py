"""
Logistic Regression — pure Python, zero external ML libraries.
Implements: sigmoid, binary cross-entropy, gradient descent,
forward pass, weight initialization.
"""
import math
import random


def sigmoid(z: float) -> float:
    """Numerically stable sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def dot(w: list[float], x: list[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))


def predict_proba(weights: list[float], bias: float, X: list[list[float]]) -> list[float]:
    """Forward pass: sigmoid(Xw + b)."""
    return [sigmoid(dot(weights, x) + bias) for x in X]


def predict(weights: list[float], bias: float, X: list[list[float]], threshold: float = 0.5) -> list[int]:
    probs = predict_proba(weights, bias, X)
    return [1 if p >= threshold else 0 for p in probs]


def binary_cross_entropy(y_true: list[int], y_pred: list[float], eps: float = 1e-9) -> float:
    """Binary cross-entropy loss."""
    n = len(y_true)
    loss = 0.0
    for yt, yp in zip(y_true, y_pred):
        yp = max(eps, min(1 - eps, yp))
        loss -= yt * math.log(yp) + (1 - yt) * math.log(1 - yp)
    return loss / n


def compute_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)


def init_weights(n_features: int, seed: int = 42) -> tuple[list[float], float]:
    """Xavier initialization."""
    random.seed(seed)
    scale = math.sqrt(2.0 / n_features)
    weights = [random.gauss(0, scale) for _ in range(n_features)]
    bias = 0.0
    return weights, bias


def compute_gradients(
    weights: list[float],
    bias: float,
    X: list[list[float]],
    y: list[int],
) -> tuple[list[float], float]:
    """
    Compute gradients of binary cross-entropy w.r.t. weights and bias.
    grad_w = (1/n) * X^T (sigmoid(Xw+b) - y)
    grad_b = (1/n) * sum(sigmoid(Xw+b) - y)
    """
    n = len(X)
    probs = predict_proba(weights, bias, X)
    errors = [p - yt for p, yt in zip(probs, y)]
    grad_w = [sum(errors[i] * X[i][j] for i in range(n)) / n for j in range(len(weights))]
    grad_b = sum(errors) / n
    return grad_w, grad_b


def sgd_step(
    weights: list[float],
    bias: float,
    grad_w: list[float],
    grad_b: float,
    lr: float,
) -> tuple[list[float], float]:
    new_w = [w - lr * gw for w, gw in zip(weights, grad_w)]
    new_b = bias - lr * grad_b
    return new_w, new_b


def train_local(
    weights: list[float],
    bias: float,
    X: list[list[float]],
    y: list[int],
    lr: float = 0.01,
    epochs: int = 5,
) -> tuple[list[float], float, float]:
    """
    Local training loop.
    Returns (new_weights, new_bias, final_loss).
    """
    for _ in range(epochs):
        grad_w, grad_b = compute_gradients(weights, bias, X, y)
        weights, bias = sgd_step(weights, bias, grad_w, grad_b, lr)
    probs = predict_proba(weights, bias, X)
    loss = binary_cross_entropy(y, probs)
    return weights, bias, loss


def l2_norm_sq(w: list[float]) -> float:
    """||w||² — squared L2 norm."""
    return sum(wi ** 2 for wi in w)


def weight_diff_norm_sq(w1: list[float], w2: list[float]) -> float:
    """||w1 - w2||²."""
    return sum((a - b) ** 2 for a, b in zip(w1, w2))

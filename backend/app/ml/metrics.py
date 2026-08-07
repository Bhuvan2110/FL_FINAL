"""
Metrics engine — pure Python, zero external ML libraries.
Implements: confusion matrix, ROC/AUC, PR curve, F1, precision,
recall, feature importance, Platt scaling.
"""
import math


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def confusion_matrix(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def precision(cm: dict) -> float:
    denom = cm["tp"] + cm["fp"]
    return cm["tp"] / denom if denom > 0 else 0.0


def recall(cm: dict) -> float:
    denom = cm["tp"] + cm["fn"]
    return cm["tp"] / denom if denom > 0 else 0.0


def f1_score(cm: dict) -> float:
    p, r = precision(cm), recall(cm)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def accuracy(cm: dict) -> float:
    total = cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]
    return (cm["tp"] + cm["tn"]) / total if total > 0 else 0.0


# ── ROC / AUC ─────────────────────────────────────────────────────────────────

def roc_curve(y_true: list[int], y_scores: list[float]) -> dict:
    """
    Compute ROC curve points.
    Returns: {"fpr": [...], "tpr": [...], "thresholds": [...], "auc": float}
    """
    thresholds = sorted(set(y_scores), reverse=True)
    fpr_list, tpr_list = [0.0], [0.0]
    P = sum(y_true)
    N = len(y_true) - P

    for t in thresholds:
        preds = [1 if s >= t else 0 for s in y_scores]
        cm = confusion_matrix(y_true, preds)
        tpr = cm["tp"] / P if P > 0 else 0.0
        fpr_val = cm["fp"] / N if N > 0 else 0.0
        fpr_list.append(fpr_val)
        tpr_list.append(tpr)

    fpr_list.append(1.0)
    tpr_list.append(1.0)

    # AUC via trapezoidal rule
    auc = sum(
        (fpr_list[i + 1] - fpr_list[i]) * (tpr_list[i + 1] + tpr_list[i]) / 2
        for i in range(len(fpr_list) - 1)
    )
    return {
        "fpr": [round(v, 4) for v in fpr_list],
        "tpr": [round(v, 4) for v in tpr_list],
        "thresholds": [round(t, 4) for t in thresholds],
        "auc": round(abs(auc), 4),
    }


# ── PR Curve ──────────────────────────────────────────────────────────────────

def pr_curve(y_true: list[int], y_scores: list[float]) -> dict:
    """Precision-Recall curve for imbalanced datasets."""
    thresholds = sorted(set(y_scores), reverse=True)
    prec_list, rec_list = [], []

    for t in thresholds:
        preds = [1 if s >= t else 0 for s in y_scores]
        cm = confusion_matrix(y_true, preds)
        prec_list.append(round(precision(cm), 4))
        rec_list.append(round(recall(cm), 4))

    return {"precision": prec_list, "recall": rec_list, "thresholds": [round(t, 4) for t in thresholds]}


# ── Feature Importance ────────────────────────────────────────────────────────

def feature_importance(weights: list[float], feature_names: list[str] = None) -> list[dict]:
    """
    Feature importance from logistic regression weights: |w_i|.
    Normalized to sum to 1.
    """
    abs_weights = [abs(w) for w in weights]
    total = sum(abs_weights) or 1.0
    names = feature_names or [f"feature_{i}" for i in range(len(weights))]
    importance = [
        {"feature": name, "importance": round(aw / total, 4), "weight": round(w, 6)}
        for name, aw, w in zip(names, abs_weights, weights)
    ]
    return sorted(importance, key=lambda x: x["importance"], reverse=True)


# ── Platt Scaling ─────────────────────────────────────────────────────────────

def platt_scaling_fit(y_true: list[int], scores: list[float], lr: float = 0.01, epochs: int = 100) -> tuple[float, float]:
    """
    Fit Platt scaling: P(y=1|s) = sigmoid(A*s + B).
    Learns A, B via gradient descent on log-loss.
    """
    A, B = 0.0, 0.0
    n = len(scores)
    for _ in range(epochs):
        dA, dB = 0.0, 0.0
        for s, yt in zip(scores, y_true):
            z = A * s + B
            if z >= 0:
                p = 1.0 / (1.0 + math.exp(-z))
            else:
                exp_z = math.exp(z)
                p = exp_z / (1.0 + exp_z)
            err = p - yt
            dA += err * s / n
            dB += err / n
        A -= lr * dA
        B -= lr * dB
    return A, B


def platt_scaling_predict(A: float, B: float, scores: list[float]) -> list[float]:
    """Apply Platt scaling to get calibrated probabilities."""
    calibrated = []
    for s in scores:
        z = A * s + B
        if z >= 0:
            p = 1.0 / (1.0 + math.exp(-z))
        else:
            exp_z = math.exp(z)
            p = exp_z / (1.0 + exp_z)
        calibrated.append(round(p, 4))
    return calibrated


# ── Full Evaluation ───────────────────────────────────────────────────────────

def full_evaluation(
    y_true: list[int],
    y_pred: list[int],
    y_scores: list[float],
    weights: list[float],
    feature_names: list[str] = None,
) -> dict:
    """Compute all metrics in one shot."""
    cm = confusion_matrix(y_true, y_pred)
    roc = roc_curve(y_true, y_scores)
    pr = pr_curve(y_true, y_scores)
    fi = feature_importance(weights, feature_names)
    return {
        "accuracy": round(accuracy(cm), 4),
        "f1": round(f1_score(cm), 4),
        "precision": round(precision(cm), 4),
        "recall": round(recall(cm), 4),
        "auc": roc["auc"],
        "confusion_matrix": cm,
        "roc_curve": roc,
        "pr_curve": pr,
        "feature_importance": fi,
    }

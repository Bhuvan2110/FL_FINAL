"""
From-scratch preprocessing: min-max normalization, column profiling,
stratified train/val/test split. Zero external ML libraries.
"""
import csv
import io
import math
import random
import hashlib
from typing import Any


# ── CSV Parsing ───────────────────────────────────────────────────────────────

def parse_csv(content: bytes) -> tuple[list[str], list[list[Any]]]:
    """Parse CSV bytes → (headers, rows)."""
    text = content.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    headers = next(reader)
    rows = [row for row in reader if any(c.strip() for c in row)]
    return headers, rows


# ── Column Profiler ───────────────────────────────────────────────────────────

def profile_columns(headers: list[str], rows: list[list[str]]) -> list[dict]:
    """
    Per-column stats: dtype, missing%, unique count, min, max, mean.
    """
    n = len(rows)
    profiles = []
    for i, h in enumerate(headers):
        vals = [r[i] if i < len(r) else "" for r in rows]
        missing = sum(1 for v in vals if v.strip() == "")
        non_empty = [v for v in vals if v.strip() != ""]
        numeric_vals = []
        for v in non_empty:
            try:
                numeric_vals.append(float(v))
            except ValueError:
                pass
        is_numeric = len(numeric_vals) == len(non_empty) and len(non_empty) > 0
        profile = {
            "column": h,
            "dtype": "numeric" if is_numeric else "categorical",
            "missing_pct": round(missing / n * 100, 2) if n > 0 else 0,
            "unique_count": len(set(non_empty)),
            "count": n,
        }
        if is_numeric and numeric_vals:
            profile["min"] = min(numeric_vals)
            profile["max"] = max(numeric_vals)
            profile["mean"] = round(sum(numeric_vals) / len(numeric_vals), 4)
            profile["std"] = round(
                math.sqrt(sum((x - profile["mean"]) ** 2 for x in numeric_vals) / len(numeric_vals)), 4
            )
        profiles.append(profile)
    return profiles


# ── Min-Max Normalization ─────────────────────────────────────────────────────

def min_max_normalize(data: list[list[float]]) -> tuple[list[list[float]], list[dict]]:
    """
    Normalize each column to [0, 1].
    Returns normalized data + scaler params (min, range per column).
    """
    if not data:
        return data, []
    n_cols = len(data[0])
    scalers = []
    for j in range(n_cols):
        col = [row[j] for row in data]
        cmin = min(col)
        cmax = max(col)
        crange = cmax - cmin if cmax != cmin else 1.0
        scalers.append({"min": cmin, "range": crange})

    normalized = []
    for row in data:
        normalized.append([(row[j] - scalers[j]["min"]) / scalers[j]["range"] for j in range(n_cols)])
    return normalized, scalers


def apply_scaler(row: list[float], scalers: list[dict]) -> list[float]:
    return [(row[j] - scalers[j]["min"]) / scalers[j]["range"] for j in range(len(row))]


# ── Stratified Split ──────────────────────────────────────────────────────────

def stratified_split(
    X: list[list[float]],
    y: list[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Stratified train/val/test split.
    Maintains class distribution across splits.
    """
    random.seed(seed)
    # Group indices by class
    class_indices: dict[int, list[int]] = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in class_indices.items():
        random.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train: n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    def extract(idx_list):
        return [X[i] for i in idx_list], [y[i] for i in idx_list]

    X_train, y_train = extract(train_idx)
    X_val, y_val = extract(val_idx)
    X_test, y_test = extract(test_idx)
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


# ── IID / Non-IID Client Partition ───────────────────────────────────────────

def partition_iid(X: list, y: list, n_clients: int, seed: int = 42) -> list[tuple]:
    """Split data evenly across clients (IID)."""
    random.seed(seed)
    indices = list(range(len(X)))
    random.shuffle(indices)
    splits = [indices[i::n_clients] for i in range(n_clients)]
    return [([X[j] for j in s], [y[j] for j in s]) for s in splits]


def partition_non_iid(X: list, y: list, n_clients: int, shards_per_client: int = 2, seed: int = 42) -> list[tuple]:
    """Non-IID: sort by label, create shards, assign 2 shards per client."""
    random.seed(seed)
    sorted_idx = sorted(range(len(y)), key=lambda i: y[i])
    n_shards = n_clients * shards_per_client
    shard_size = max(1, len(sorted_idx) // n_shards)
    shards = [sorted_idx[i * shard_size:(i + 1) * shard_size] for i in range(n_shards)]
    random.shuffle(shards)
    clients = []
    for c in range(n_clients):
        client_idx = []
        for s in range(shards_per_client):
            client_idx.extend(shards[c * shards_per_client + s])
        clients.append(([X[j] for j in client_idx], [y[j] for j in client_idx]))
    return clients


# ── Data Fingerprint ──────────────────────────────────────────────────────────

def compute_input_hash(row_dict: dict) -> str:
    """Stable hash for a prediction input row."""
    canonical = str(sorted(row_dict.items()))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]

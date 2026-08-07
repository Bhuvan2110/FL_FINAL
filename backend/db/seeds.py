"""
Seed script: demo users, sample dataset (Pima Indians Diabetes), pre-run experiments.
Run: python -m backend.db.seeds
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import csv, io, json, random
from app.db.supabase_client import get_supabase
from app.ml.preprocessing import min_max_normalize, stratified_split, partition_iid
from app.ml.fl_algorithms import run_fedavg, run_fedprox, run_scaffold, run_central
from app.ml.differential_privacy import run_dpsgd
from app.ml.logistic_regression import predict, predict_proba
from app.ml.metrics import full_evaluation


SUPER_ADMIN_EMAIL = "sbhuvan847@gmail.com"

# ── Pima Indians Diabetes-like synthetic dataset ──────────────────────────────
def generate_diabetes_dataset(n=500, seed=42):
    random.seed(seed)
    rows = []
    headers = ["pregnancies","glucose","blood_pressure","skin_thickness","insulin","bmi","dpf","age","outcome"]
    for _ in range(n):
        glucose = random.gauss(120, 30)
        bmi = random.gauss(32, 7)
        age = random.randint(21, 80)
        label = 1 if (glucose > 140 or bmi > 35) and random.random() > 0.3 else 0
        rows.append([
            random.randint(0, 15),
            round(max(50, glucose), 1),
            round(random.gauss(72, 12), 1),
            round(random.gauss(29, 10), 1),
            round(max(0, random.gauss(120, 80)), 1),
            round(max(15, bmi), 1),
            round(random.uniform(0.08, 2.5), 3),
            age,
            label,
        ])
    return headers, rows


def seed():
    sb = get_supabase()
    print("🌱 Seeding FL Platform…")

    # Generate CSV
    headers, rows = generate_diabetes_dataset(500)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    w.writerows(rows)
    csv_content = buf.getvalue().encode()

    # Upload to storage
    storage_path = "datasets/seed/diabetes_synthetic.csv"
    try:
        sb.storage.from_("datasets").upload(storage_path, csv_content, {"content-type": "text/csv", "upsert": "true"})
        print("  ✓ Dataset uploaded to Supabase Storage")
    except Exception as e:
        print(f"  ⚠ Storage upload: {e}")

    # Insert dataset record
    from app.ml.preprocessing import parse_csv, profile_columns
    hdr, data_rows = parse_csv(csv_content)
    profiles = profile_columns(hdr, data_rows)
    ds_result = sb.table("datasets").insert({
        "user_id": None,  # seed data
        "filename": "diabetes_synthetic.csv",
        "storage_path": storage_path,
        "cols": profiles,
        "row_count": len(data_rows),
    }).execute()
    dataset_id = ds_result.data[0]["id"]
    print(f"  ✓ Dataset record: {dataset_id}")

    # Prepare ML data
    X_raw = [[float(r[j]) for j in range(len(headers)-1)] for r in rows]
    y = [int(r[-1]) for r in rows]
    X, scalers = min_max_normalize(X_raw)
    splits = stratified_split(X, y)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    feature_names = headers[:-1]

    # Run all 5 algorithms and store
    configs = [
        ("fedavg",   lambda: run_fedavg(partition_iid(X_train, y_train, 5), 20, 0.05, 5, X_val, y_val)),
        ("fedprox",  lambda: run_fedprox(partition_iid(X_train, y_train, 5), 20, 0.05, 0.1, 5, X_val, y_val)),
        ("scaffold", lambda: run_scaffold(partition_iid(X_train, y_train, 5), 20, 0.05, 5, X_val, y_val)),
        ("central",  lambda: run_central(X_train, y_train, 20, 0.05, 5, X_val, y_val)),
    ]

    for algo, runner in configs:
        print(f"  → Training {algo}…", end=' ', flush=True)
        result = runner()
        history, weights, bias = result[0], result[1], result[2]

        exp = sb.table("experiments").insert({
            "dataset_id": dataset_id,
            "algorithm": algo,
            "status": "completed",
            "hyperparams": {"n_rounds": 20, "lr": 0.05, "n_clients": 5, "local_epochs": 5},
        }).execute()
        exp_id = exp.data[0]["id"]

        for rd in history:
            sb.table("rounds").insert({
                "experiment_id": exp_id,
                "round_num": rd["round"],
                "loss": rd.get("loss"),
                "accuracy": rd.get("accuracy"),
                "val_accuracy": rd.get("val_accuracy"),
                "num_clients": rd.get("num_clients", 1),
            }).execute()

        preds = predict(weights, bias, X_test)
        scores = predict_proba(weights, bias, X_test)
        eval_r = full_evaluation(y_test, preds, scores, weights, feature_names)

        sb.table("metrics").insert({
            "experiment_id": exp_id,
            "accuracy": eval_r["accuracy"],
            "f1": eval_r["f1"],
            "auc": eval_r["auc"],
            "precision_score": eval_r["precision"],
            "recall": eval_r["recall"],
            "confusion_matrix": eval_r["confusion_matrix"],
            "roc_curve": eval_r["roc_curve"],
            "feature_importance": eval_r["feature_importance"],
        }).execute()

        model_data = json.dumps({"weights": weights, "bias": bias, "scalers": scalers, "feature_names": feature_names})
        model_path = f"models/{exp_id}/model_v1.json"
        try:
            sb.storage.from_("models").upload(model_path, model_data.encode(), {"content-type": "application/json", "upsert": "true"})
        except Exception:
            pass
        sb.table("models").insert({"experiment_id": exp_id, "weights_path": model_path, "version": 1}).execute()
        print(f"acc={eval_r['accuracy']:.3f} f1={eval_r['f1']:.3f}")

    # DP-SGD
    print("  → Training dpsgd…", end=' ', flush=True)
    history, weights, bias, pb = run_dpsgd(partition_iid(X_train, y_train, 5), 20, 0.05, 5, 1.0, 1.0, 1e-5, 32, X_val, y_val)
    exp = sb.table("experiments").insert({
        "dataset_id": dataset_id,
        "algorithm": "dpsgd",
        "status": "completed",
        "hyperparams": {"n_rounds": 20, "lr": 0.05, "clip_norm": 1.0, "noise_multiplier": 1.0},
    }).execute()
    exp_id = exp.data[0]["id"]
    for rd in history:
        sb.table("rounds").insert({"experiment_id": exp_id, "round_num": rd["round"], "loss": rd.get("loss"), "accuracy": rd.get("accuracy"), "val_accuracy": rd.get("val_accuracy"), "num_clients": rd.get("num_clients", 5)}).execute()
    for p in pb:
        sb.table("privacy_budget").insert({"experiment_id": exp_id, "round_num": p["round"], "epsilon": p["epsilon"], "delta": p["delta"], "noise_multiplier": p["noise_multiplier"], "clip_norm": p["clip_norm"]}).execute()
    preds = predict(weights, bias, X_test)
    scores = predict_proba(weights, bias, X_test)
    eval_r = full_evaluation(y_test, preds, scores, weights, feature_names)
    sb.table("metrics").insert({"experiment_id": exp_id, "accuracy": eval_r["accuracy"], "f1": eval_r["f1"], "auc": eval_r["auc"], "precision_score": eval_r["precision"], "recall": eval_r["recall"], "confusion_matrix": eval_r["confusion_matrix"], "roc_curve": eval_r["roc_curve"], "feature_importance": eval_r["feature_importance"]}).execute()
    print(f"acc={eval_r['accuracy']:.3f} ε={pb[-1]['epsilon']:.3f}")

    print("\n✅ Seed complete! All 5 models seeded.")
    print(f"   Super admin: {SUPER_ADMIN_EMAIL}")


if __name__ == "__main__":
    seed()

"""
Celery app + Redis broker + FL training tasks.
"""
import json
from celery import Celery
from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "fl_platform",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)


@celery_app.task(bind=True, name="tasks.run_training")
def run_training_task(self, experiment_id: str, config: dict):
    """
    Async Celery task for FL/Central training.
    Updates experiment status and pushes round metrics to Supabase.
    """
    from app.db.supabase_client import get_supabase
    from app.ml.preprocessing import (
        parse_csv, min_max_normalize, stratified_split,
        partition_iid, partition_non_iid,
    )
    from app.ml.fl_algorithms import run_fedavg, run_fedprox, run_scaffold, run_central
    from app.ml.differential_privacy import run_dpsgd
    from app.ml.logistic_regression import predict, predict_proba
    from app.ml.metrics import full_evaluation

    sb = get_supabase()
    algorithm = config.get("algorithm", "fedavg")
    dataset_id = config.get("dataset_id")
    n_rounds = config.get("n_rounds", 20)
    lr = config.get("lr", 0.01)
    n_clients = config.get("n_clients", 5)
    local_epochs = config.get("local_epochs", 5)
    mu = config.get("mu", 0.1)
    clip_norm = config.get("clip_norm", 1.0)
    noise_multiplier = config.get("noise_multiplier", 1.0)
    delta = config.get("delta", 1e-5)
    non_iid = config.get("non_iid", False)

    try:
        # Update status to running
        sb.table("experiments").update({"status": "running", "celery_task_id": self.request.id}).eq("id", experiment_id).execute()

        # Load dataset from Supabase Storage
        ds_row = sb.table("datasets").select("*").eq("id", dataset_id).single().execute().data
        storage_path = ds_row["storage_path"]
        csv_bytes = sb.storage.from_("datasets").download(storage_path)
        headers, rows = parse_csv(csv_bytes)

        # Ensure sl.no is present as a feature (excluding the target column which is the last one)
        has_sl_no = any(h.lower().replace(" ", "").replace("_", "").replace(".", "") in ["slno", "sno", "serialno"] for h in headers[:-1])
        if not has_sl_no:
            headers.insert(0, "sl.no")
            for idx, r in enumerate(rows):
                r.insert(0, str(idx + 1))

        # Determine if columns are numeric or categorical
        encoders = {}
        X_raw = []
        y = []
        
        # Parse target column (last column)
        target_col = len(headers) - 1
        try:
            y = [int(float(r[target_col])) for r in rows]
        except (ValueError, IndexError):
            raise ValueError("CSV must contain a binary target (0 or 1) in the last column")

        # Parse feature columns
        feature_indices = [j for j in range(len(headers)) if j != target_col]
        feature_names = [headers[j] for j in feature_indices]
        
        # Profile columns to see which are categorical
        for idx in feature_indices:
            col_name = headers[idx]
            col_vals = [r[idx] for r in rows]
            
            # Check if all non-empty values are numeric
            is_numeric = True
            for v in col_vals:
                if v.strip() == "":
                    continue
                try:
                    float(v)
                except ValueError:
                    is_numeric = False
                    break
            
            if not is_numeric:
                # Collect unique non-empty categories
                unique_cats = sorted(list(set(v.strip() for v in col_vals if v.strip() != "")))
                encoders[col_name] = unique_cats
        
        # Now construct X_raw with encoding applied
        for r in rows:
            row_features = []
            for idx in feature_indices:
                col_name = headers[idx]
                val_str = r[idx].strip() if idx < len(r) else ""
                
                if col_name in encoders:
                    # Categorical column
                    if val_str in encoders[col_name]:
                        val_float = float(encoders[col_name].index(val_str))
                    else:
                        val_float = 0.0  # default for missing/unknown categories
                else:
                    # Numeric column
                    try:
                        val_float = float(val_str) if val_str != "" else 0.0
                    except ValueError:
                        val_float = 0.0
                
                row_features.append(val_float)
            X_raw.append(row_features)

        X, scalers = min_max_normalize(X_raw)
        splits = stratified_split(X, y, train_ratio=0.70, val_ratio=0.15)
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]

        # Partition clients
        if algorithm != "central":
            if non_iid:
                clients = partition_non_iid(X_train, y_train, n_clients)
            else:
                clients = partition_iid(X_train, y_train, n_clients)

        # Run algorithm
        privacy_history = []
        if algorithm == "fedavg":
            history, weights, bias = run_fedavg(clients, n_rounds, lr, local_epochs, X_val, y_val)
        elif algorithm == "fedprox":
            history, weights, bias = run_fedprox(clients, n_rounds, lr, mu, local_epochs, X_val, y_val)
        elif algorithm == "scaffold":
            history, weights, bias = run_scaffold(clients, n_rounds, lr, local_epochs, X_val, y_val)
        elif algorithm == "dpsgd":
            history, weights, bias, privacy_history = run_dpsgd(
                clients, n_rounds, lr, local_epochs, clip_norm, noise_multiplier, delta, 32, X_val, y_val
            )
        elif algorithm == "central":
            history, weights, bias = run_central(X_train, y_train, n_rounds, lr, local_epochs, X_val, y_val)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Store round metrics in Supabase
        for rd in history:
            sb.table("rounds").insert({
                "experiment_id": experiment_id,
                "round_num": rd["round"],
                "loss": rd.get("loss"),
                "accuracy": rd.get("accuracy"),
                "val_loss": rd.get("val_loss"),
                "val_accuracy": rd.get("val_accuracy"),
                "num_clients": rd.get("num_clients", 1),
            }).execute()

        # Store privacy budget (DP-SGD only)
        for pb in privacy_history:
            sb.table("privacy_budget").insert({
                "experiment_id": experiment_id,
                "round_num": pb["round"],
                "epsilon": pb["epsilon"],
                "delta": pb["delta"],
                "noise_multiplier": pb.get("noise_multiplier"),
                "clip_norm": pb.get("clip_norm"),
            }).execute()

        # Evaluate on test set
        preds = predict(weights, bias, X_test)
        scores = predict_proba(weights, bias, X_test)
        eval_result = full_evaluation(y_test, preds, scores, weights, feature_names)

        # Store model weights
        model_data = json.dumps({
            "weights": weights,
            "bias": bias,
            "scalers": scalers,
            "feature_names": feature_names,
            "encoders": encoders
        })
        model_path = f"models/{experiment_id}/model_v1.json"
        sb.storage.from_("models").upload(model_path, model_data.encode(), {"content-type": "application/json"})

        # Store model record
        sb.table("models").insert({
            "experiment_id": experiment_id,
            "weights_path": model_path,
            "version": 1,
        }).execute()

        # Store metrics
        sb.table("metrics").insert({
            "experiment_id": experiment_id,
            "accuracy": eval_result["accuracy"],
            "f1": eval_result["f1"],
            "auc": eval_result["auc"],
            "precision_score": eval_result["precision"],
            "recall": eval_result["recall"],
            "confusion_matrix": eval_result["confusion_matrix"],
            "roc_curve": eval_result["roc_curve"],
            "feature_importance": eval_result["feature_importance"],
        }).execute()

        # Mark complete
        sb.table("experiments").update({"status": "completed"}).eq("id", experiment_id).execute()
        return {"status": "completed", "experiment_id": experiment_id, "metrics": eval_result}

    except Exception as e:
        sb.table("experiments").update({"status": "failed"}).eq("id", experiment_id).execute()
        raise e

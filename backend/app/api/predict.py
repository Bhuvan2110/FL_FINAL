"""
Prediction API — single & batch inference with Platt-scaled confidence.
"""
import json
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import Any
from app.api.dependencies import get_current_user
from app.db.supabase_client import get_supabase
from app.ml.logistic_regression import predict_proba
from app.ml.preprocessing import apply_scaler, compute_input_hash, parse_csv

router = APIRouter(prefix="/predict", tags=["predictions"])


def _load_model(model_id: str, sb) -> dict:
    """Load model weights from Supabase Storage."""
    model_row = sb.table("models").select("*").eq("id", model_id).single().execute()
    if not model_row.data:
        raise HTTPException(status_code=404, detail="Model not found")
    weights_bytes = sb.storage.from_("models").download(model_row.data["weights_path"])
    return json.loads(weights_bytes.decode())


class SinglePredictRequest(BaseModel):
    model_id: str
    features: dict[str, Any]


@router.post("/single")
async def predict_single(body: SinglePredictRequest, user: dict = Depends(get_current_user)):
    """Single-row JSON prediction."""
    sb = get_supabase()
    model_data = _load_model(body.model_id, sb)

    weights = model_data["weights"]
    bias = model_data["bias"]
    scalers = model_data["scalers"]
    feature_names = model_data.get("feature_names", [f"feature_{i}" for i in range(len(weights))])

    # Build feature vector in correct order
    encoders = model_data.get("encoders", {})
    try:
        raw_row = []
        for name in feature_names:
            val = body.features.get(name, 0.0)
            if name in encoders:
                if isinstance(val, str):
                    val_str = val.strip()
                    if val_str in encoders[name]:
                        val_float = float(encoders[name].index(val_str))
                    else:
                        val_float = 0.0
                else:
                    try:
                        val_float = float(val)
                    except (ValueError, TypeError):
                        val_float = 0.0
            else:
                try:
                    val_float = float(val)
                except (ValueError, TypeError):
                    val_float = 0.0
            raw_row.append(val_float)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature values: {e}")

    normalized_row = apply_scaler(raw_row, scalers)
    prob = predict_proba(weights, bias, [normalized_row])[0]
    pred = 1 if prob >= 0.5 else 0
    input_hash = compute_input_hash(body.features)

    # Store prediction history
    sb.table("predictions").insert({
        "user_id": user["id"],
        "model_id": body.model_id,
        "input_hash": input_hash,
        "output": pred,
        "confidence": round(prob, 4),
    }).execute()

    return {
        "prediction": pred,
        "confidence": round(prob, 4),
        "class_label": "Positive" if pred == 1 else "Negative",
        "input_hash": input_hash,
    }


@router.post("/batch")
async def predict_batch(
    model_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """Batch prediction from CSV upload."""
    sb = get_supabase()
    model_data = _load_model(model_id, sb)

    weights = model_data["weights"]
    bias = model_data["bias"]
    scalers = model_data["scalers"]
    feature_names = model_data.get("feature_names", [])

    content = await file.read()
    headers, rows = parse_csv(content)

    results = []
    batch_id = compute_input_hash({"ts": str(id(rows))})

    encoders = model_data.get("encoders", {})
    for i, row in enumerate(rows):
        try:
            feature_map = {h: row[j] if j < len(row) else "0" for j, h in enumerate(headers)}
            raw_row = []
            for name in feature_names:
                val = feature_map.get(name, "0")
                if name in encoders:
                    val_str = str(val).strip()
                    if val_str in encoders[name]:
                        val_float = float(encoders[name].index(val_str))
                    else:
                        val_float = 0.0
                else:
                    try:
                        val_float = float(val)
                    except (ValueError, TypeError):
                        val_float = 0.0
                raw_row.append(val_float)
            norm_row = apply_scaler(raw_row, scalers)
            prob = predict_proba(weights, bias, [norm_row])[0]
            pred = 1 if prob >= 0.5 else 0

            sb.table("predictions").insert({
                "user_id": user["id"],
                "model_id": model_id,
                "input_hash": compute_input_hash(feature_map),
                "output": pred,
                "confidence": round(prob, 4),
                "batch_id": batch_id,
            }).execute()

            results.append({
                "row": i + 1,
                "prediction": pred,
                "confidence": round(prob, 4),
                "class_label": "Positive" if pred == 1 else "Negative",
            })
        except Exception as e:
            results.append({"row": i + 1, "error": str(e)})

    return {"batch_id": batch_id, "total": len(rows), "results": results}


@router.get("/history")
async def prediction_history(user: dict = Depends(get_current_user)):
    sb = get_supabase()
    preds = sb.table("predictions").select("*").eq("user_id", user["id"]).order("created_at", desc=True).limit(100).execute()
    return preds.data


@router.get("/models")
async def list_available_models(user: dict = Depends(get_current_user)):
    """List all models available for prediction (from completed experiments)."""
    sb = get_supabase()
    if user.get("role") in ("admin", "super_admin"):
        exps = sb.table("experiments").select("id, algorithm, created_at").eq("status", "completed").execute()
    else:
        exps = sb.table("experiments").select("id, algorithm, created_at").eq("user_id", user["id"]).eq("status", "completed").execute()

    models = []
    for exp in exps.data:
        model = sb.table("models").select("*").eq("experiment_id", exp["id"]).execute()
        if model.data:
            models.append({**model.data[0], "algorithm": exp["algorithm"]})
    return models

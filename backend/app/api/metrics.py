"""
Metrics API — comparison endpoint, privacy-utility chart data.
"""
from fastapi import APIRouter, Depends, Query
from app.api.dependencies import get_current_user
from app.db.supabase_client import get_supabase

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/compare")
async def compare_metrics(
    models: str = Query(default="fedavg,fedprox,scaffold,dpsgd,central"),
    user: dict = Depends(get_current_user),
):
    """
    GET /metrics/compare?models=fedavg,fedprox,scaffold,dpsgd,central
    Returns per-model accuracy, F1, AUC, epsilon for charts.
    """
    requested = [m.strip() for m in models.split(",")]
    sb = get_supabase()

    if user.get("role") in ("admin", "super_admin"):
        exps = sb.table("experiments").select("id, algorithm, status").eq("status", "completed").execute()
    else:
        exps = sb.table("experiments").select("id, algorithm, status").eq("user_id", user["id"]).eq("status", "completed").execute()

    comparison = []
    for exp in exps.data:
        if exp["algorithm"] not in requested:
            continue
        m = sb.table("metrics").select("accuracy,f1,auc,precision_score,recall,confusion_matrix,roc_curve,feature_importance").eq("experiment_id", exp["id"]).single().execute()
        pb = sb.table("privacy_budget").select("epsilon,round_num").eq("experiment_id", exp["id"]).order("round_num", desc=True).limit(1).execute()
        if m.data:
            comparison.append({
                "experiment_id": exp["id"],
                "algorithm": exp["algorithm"],
                **m.data,
                "final_epsilon": pb.data[0]["epsilon"] if pb.data else None,
            })

    return comparison


@router.get("/privacy-utility")
async def privacy_utility_curve(user: dict = Depends(get_current_user)):
    """
    Privacy-utility tradeoff: accuracy vs epsilon across DP-SGD runs.
    """
    sb = get_supabase()

    if user.get("role") in ("admin", "super_admin"):
        exps = sb.table("experiments").select("id").eq("algorithm", "dpsgd").eq("status", "completed").execute()
    else:
        exps = sb.table("experiments").select("id").eq("user_id", user["id"]).eq("algorithm", "dpsgd").eq("status", "completed").execute()

    curve_data = []
    for exp in exps.data:
        rounds = sb.table("rounds").select("round_num, accuracy, val_accuracy").eq("experiment_id", exp["id"]).order("round_num").execute()
        pb = sb.table("privacy_budget").select("round_num, epsilon").eq("experiment_id", exp["id"]).order("round_num").execute()
        # Merge by round_num
        pb_map = {p["round_num"]: p["epsilon"] for p in pb.data}
        for r in rounds.data:
            eps = pb_map.get(r["round_num"])
            if eps is not None:
                curve_data.append({
                    "experiment_id": exp["id"],
                    "round": r["round_num"],
                    "accuracy": r.get("val_accuracy") or r.get("accuracy"),
                    "epsilon": eps,
                })
    return curve_data


@router.get("/experiment/{experiment_id}")
async def get_experiment_metrics(experiment_id: str, user: dict = Depends(get_current_user)):
    """Full metrics for a single experiment."""
    sb = get_supabase()
    metrics = sb.table("metrics").select("*").eq("experiment_id", experiment_id).single().execute()
    rounds = sb.table("rounds").select("*").eq("experiment_id", experiment_id).order("round_num").execute()
    pb = sb.table("privacy_budget").select("*").eq("experiment_id", experiment_id).order("round_num").execute()
    return {
        "metrics": metrics.data,
        "rounds": rounds.data,
        "privacy_budget": pb.data,
    }

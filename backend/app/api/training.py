"""
Training API — trigger FL experiments, monitor status, compare results.
"""
import uuid
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from app.api.dependencies import get_current_user
from app.db.supabase_client import get_supabase

router = APIRouter(prefix="/training", tags=["training"])


class TrainingConfig(BaseModel):
    dataset_id: str
    algorithm: str  # fedavg | fedprox | scaffold | dpsgd | central
    n_rounds: int = 20
    lr: float = 0.01
    n_clients: int = 5
    local_epochs: int = 5
    mu: float = 0.1          # FedProx proximal term
    clip_norm: float = 1.0   # DP-SGD
    noise_multiplier: float = 1.0  # DP-SGD
    delta: float = 1e-5      # DP-SGD
    non_iid: bool = False


VALID_ALGORITHMS = {"fedavg", "fedprox", "scaffold", "dpsgd", "central"}


@router.post("/start")
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    if config.algorithm not in VALID_ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Algorithm must be one of {VALID_ALGORITHMS}")

    sb = get_supabase()
    # Verify dataset belongs to user
    ds = sb.table("datasets").select("id").eq("id", config.dataset_id).eq("user_id", user["id"]).execute()
    if not ds.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Create experiment record
    exp = sb.table("experiments").insert({
        "user_id": user["id"],
        "dataset_id": config.dataset_id,
        "algorithm": config.algorithm,
        "status": "pending",
        "hyperparams": config.model_dump(),
    }).execute()
    experiment_id = exp.data[0]["id"]

    # Dispatch task: check if Redis is running first to avoid Celery hangs
    import socket
    from urllib.parse import urlparse
    from app.core.config import get_settings
    
    settings = get_settings()
    redis_available = False
    try:
        r_uri = urlparse(settings.redis_url)
        host = r_uri.hostname or "localhost"
        port = r_uri.port or 6379
        s = socket.create_connection((host, port), timeout=0.8)
        s.close()
        redis_available = True
    except Exception:
        pass

    if redis_available:
        try:
            from app.tasks.celery_app import run_training_task
            task = run_training_task.delay(experiment_id, {**config.model_dump(), "user_id": user["id"]})
            sb.table("experiments").update({"celery_task_id": task.id}).eq("id", experiment_id).execute()
            return {"experiment_id": experiment_id, "task_id": task.id, "status": "pending"}
        except Exception as e:
            print(f"⚠️ Celery dispatch failed despite Redis online ({str(e)}). Falling back to background thread.")
            redis_available = False

    if not redis_available:
        # Fallback: run in a background thread if Celery/Redis is not available
        print("⚠️ Celery/Redis is offline. Running training task in local background thread.")
        from app.tasks.celery_app import run_training_task
        
        class DummyCeleryTask:
            class DummyRequest:
                def __init__(self):
                    self.id = f"local-{uuid.uuid4()}"
            def __init__(self):
                self.request = self.DummyRequest()
                
        dummy_task = DummyCeleryTask()
        background_tasks.add_task(
            run_training_task.run.__func__,
            dummy_task,
            experiment_id,
            {**config.model_dump(), "user_id": user["id"]}
        )
        sb.table("experiments").update({"celery_task_id": dummy_task.request.id}).eq("id", experiment_id).execute()
        return {"experiment_id": experiment_id, "task_id": dummy_task.request.id, "status": "pending"}


@router.get("/list")
async def list_experiments(user: dict = Depends(get_current_user)):
    sb = get_supabase()
    exps = sb.table("experiments").select("*, datasets(filename)").eq("user_id", user["id"]).order("created_at", desc=True).execute()
    return exps.data


@router.get("/compare")
async def compare_experiments(user: dict = Depends(get_current_user)):
    """Return metrics for all completed experiments for comparison charts."""
    sb = get_supabase()

    if user.get("role") in ("admin", "super_admin"):
        exps = sb.table("experiments").select("id, algorithm, status, created_at, datasets(filename)").eq("status", "completed").execute()
    else:
        exps = sb.table("experiments").select("id, algorithm, status, created_at, datasets(filename)").eq("user_id", user["id"]).eq("status", "completed").execute()

    results = []
    for exp in exps.data:
        metrics = sb.table("metrics").select("*").eq("experiment_id", exp["id"]).single().execute()
        rounds = sb.table("rounds").select("round_num, loss, accuracy, val_accuracy").eq("experiment_id", exp["id"]).order("round_num").execute()
        pb = sb.table("privacy_budget").select("round_num, epsilon").eq("experiment_id", exp["id"]).order("round_num").execute()
        if metrics.data:
            results.append({
                "experiment_id": exp["id"],
                "algorithm": exp["algorithm"],
                "created_at": exp.get("created_at"),
                "dataset_filename": exp.get("datasets", {}).get("filename") if exp.get("datasets") else None,
                "metrics": metrics.data,
                "rounds": rounds.data,
                "privacy_budget": pb.data,
            })

    return results


@router.get("/{experiment_id}/status")
async def get_training_status(experiment_id: str, user: dict = Depends(get_current_user)):
    sb = get_supabase()
    exp = sb.table("experiments").select("*, datasets(filename)").eq("id", experiment_id).single().execute()
    if not exp.data:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Get rounds
    rounds = sb.table("rounds").select("*").eq("experiment_id", experiment_id).order("round_num").execute()

    # Get metrics if completed
    metrics_data = None
    privacy_data = []
    if exp.data.get("status") == "completed":
        try:
            metrics = sb.table("metrics").select("*").eq("experiment_id", experiment_id).single().execute()
            metrics_data = metrics.data
        except Exception:
            pass
        pb = sb.table("privacy_budget").select("round_num, epsilon, delta").eq("experiment_id", experiment_id).order("round_num").execute()
        privacy_data = pb.data or []

    return {
        "experiment": exp.data,
        "rounds": rounds.data,
        "latest_round": rounds.data[-1] if rounds.data else None,
        "metrics": metrics_data,
        "privacy_budget": privacy_data,
    }


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: str, user: dict = Depends(get_current_user)):
    sb = get_supabase()

    # Verify experiment exists
    exp = sb.table("experiments").select("id, user_id").eq("id", experiment_id).single().execute()
    if not exp.data:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Allow owner or admin to delete
    is_admin = user.get("role") in ("admin", "super_admin")
    if exp.data["user_id"] != user["id"] and not is_admin:
        raise HTTPException(status_code=403, detail="Not authorized to delete this experiment")

    # CASCADE in DB schema handles rounds, metrics, models, privacy_budget
    sb.table("experiments").delete().eq("id", experiment_id).execute()
    return {"message": "Experiment deleted"}

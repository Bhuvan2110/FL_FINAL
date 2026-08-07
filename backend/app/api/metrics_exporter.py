"""
Prometheus metrics exporter.
Exposes /metrics endpoint with FL-specific gauges and counters.
"""
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response

router = APIRouter(tags=["observability"])

# ── Prometheus Metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter("fl_api_requests_total", "Total API requests", ["method", "endpoint", "status"])
TRAINING_JOBS = Gauge("fl_training_jobs_active", "Currently running training jobs")
TRAINING_DURATION = Histogram("fl_training_duration_seconds", "Training job duration in seconds")
ROUNDS_COMPLETED = Counter("fl_rounds_completed_total", "Total training rounds completed", ["algorithm"])
ACCURACY_GAUGE = Gauge("fl_latest_accuracy", "Latest model accuracy", ["algorithm"])
PREDICTIONS_COUNT = Counter("fl_predictions_total", "Total predictions served", ["type"])


@router.get("/metrics")
async def metrics():
    """Prometheus /metrics scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Middleware helper ─────────────────────────────────────────────────────────
def record_request(method: str, endpoint: str, status: int):
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()


def record_round(algorithm: str, accuracy: float):
    ROUNDS_COMPLETED.labels(algorithm=algorithm).inc()
    ACCURACY_GAUGE.labels(algorithm=algorithm).set(accuracy)


def record_prediction(prediction_type: str = "single"):
    PREDICTIONS_COUNT.labels(type=prediction_type).inc()

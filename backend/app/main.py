"""
FastAPI application entry point.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.api import auth, datasets, training, predict, metrics, metrics_exporter, agent

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: configure MLflow (optional — skipped if mlflow not installed/offline)
    import socket
    from urllib.parse import urlparse
    try:
        # Fast reachability check for MLflow tracking URI
        uri = urlparse(settings.mlflow_tracking_uri)
        host = uri.hostname or "localhost"
        port = uri.port or 5000
        
        # Test connection with 1-second timeout
        s = socket.create_connection((host, port), timeout=1.0)
        s.close()
        
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("fl-platform")
        print("✅ Connected to MLflow tracking server.")
    except Exception:
        print("⚠️ MLflow server is offline. Skipping MLflow tracking.")
    yield
    # Shutdown: nothing needed


app = FastAPI(
    title="FL Platform API",
    description="Federated Learning for Privacy-Preserving Predictive Analytics",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

cors_allowed_origins = list(dict.fromkeys([
    "https://fedlearn-os.web.app",
    "https://fedlearn-os.firebaseapp.com",
    "https://fl-platform-ui-8vqt.onrender.com",
    "http://localhost:5173",
    "http://localhost:3000",
] + settings.cors_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def cors_and_security_middleware(request: Request, call_next):
    origin = request.headers.get("origin", "*")
    if request.method == "OPTIONS":
        from fastapi.responses import Response
        res = Response(status_code=204)
        res.headers["Access-Control-Allow-Origin"] = origin if origin != "*" else "*"
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        res.headers["Access-Control-Allow-Headers"] = "*"
        res.headers["Access-Control-Allow-Credentials"] = "true"
        return res

    response = await call_next(request)
    if origin and origin != "*":
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    else:
        response.headers["Access-Control-Allow-Origin"] = "*"

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(predict.router)
app.include_router(metrics.router)
app.include_router(metrics_exporter.router)
app.include_router(agent.router)


@app.get("/")
async def root():
    return {
        "message": "FL Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "algorithms": ["fedavg", "fedprox", "scaffold", "dpsgd", "central"],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "environment": settings.environment}

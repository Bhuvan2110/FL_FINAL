# FL Platform — Federated Learning for Privacy-Preserving Predictive Analytics

> Full-stack research platform implementing 5 ML algorithms from mathematical first principles.
> **Zero external ML libraries** — pure Python for all FL algorithms.

## Live Deployment

- **Frontend App (Web UI)**: [https://fl-platform-ui-8vqt.onrender.com](https://fl-platform-ui-8vqt.onrender.com)
- **Backend API Documentation**: [https://fl-platform-api-kg2m.onrender.com/docs](https://fl-platform-api-kg2m.onrender.com/docs)

## Tech Stack

| Layer | Technology |
|---|---|
| Database | Supabase PostgreSQL + RLS |
| Storage | Supabase Storage (CSV datasets, model weights) |
| Realtime | Supabase Realtime WebSocket (live training charts) |
| Auth | Supabase Auth + JWT + 3-tier RBAC |
| Backend | FastAPI + Python 3.11 |
| Async | Celery + Redis |
| ML Engine | Pure Python (FedAvg, FedProx, SCAFFOLD, DP-SGD, Central) |
| Encryption | AES-256-GCM + RSA-2048-OAEP hybrid |
| Frontend | React + Vite + Tailwind CSS |
| Charts | Recharts (ROC, convergence, feature importance) |
| Tracking | MLflow |
| Metrics | Prometheus + Grafana Cloud |
| CI/CD | GitHub Actions → Render |

## Models

| Model | Accuracy | Privacy | Non-IID |
|---|---|---|---|
| Central Training | ~95% | None | N/A |
| FedAvg | ~87% | No DP | Degrades |
| FedProx (μ=0.1) | ~89% | No DP | Better |
| SCAFFOLD | ~91% | No DP | Best FL |
| FL + DP-SGD | ~80% | ε tracked | Moderate |

## Quick Start — Local Development

### 1. Clone and set up environment

```bash
git clone https://github.com/your-repo/fl-platform.git
cd fl-platform
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
# Edit both .env files with your Supabase credentials
```

### 2. Run Supabase schema

Open Supabase SQL Editor and run `backend/app/db/schema.sql`

### 3. Start with Docker Compose

```bash
docker-compose up
```

Or start services individually:

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Celery worker (separate terminal)
celery -A app.tasks.celery_app.celery_app worker --loglevel=info

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

### 4. Seed demo data

```bash
cd backend && python -m db.seeds
```

### 5. Open the app

- Frontend: http://localhost:5173
- API docs: http://localhost:8000/docs
- MLflow: http://localhost:5000

## API Endpoints

```
POST   /auth/login                        — Sign in
POST   /auth/signup                       — Register
GET    /auth/me                           — Current user
GET    /auth/security/public-key          — Server RSA public key

POST   /dataset/upload                    — Upload CSV
GET    /dataset/preview/{id}              — Column profiler + preview
GET    /dataset/list                      — List datasets

POST   /training/start                    — Start FL training
GET    /training/{id}/status              — Status + rounds
GET    /training/compare                  — All completed experiments
GET    /training/list                     — List experiments

POST   /predict/single                    — Single JSON prediction
POST   /predict/batch                     — Batch CSV prediction
GET    /predict/history                   — Prediction history
GET    /predict/models                    — Available models

GET    /metrics/compare?models=...        — Model comparison data
GET    /metrics/privacy-utility           — Privacy-utility curve (DP-SGD)

GET    /metrics                           — Prometheus scrape endpoint
GET    /health                            — Health check
```

## Running Tests

```bash
cd backend
pytest tests/ --cov=app --cov-report=term-missing -v
# Target: ≥85% coverage
```

## Deployment — Render

The project is hosted on Render via Blueprint:
- **Frontend App (Web UI)**: [https://fl-platform-ui-8vqt.onrender.com](https://fl-platform-ui-8vqt.onrender.com)
- **Backend API**: [https://fl-platform-api-kg2m.onrender.com](https://fl-platform-api-kg2m.onrender.com)

1. Push to GitHub
2. Connect repo to Render
3. `render.yaml` auto-configures all services
4. Set env vars in Render dashboard
5. GitHub Actions deploys on push to `main`


## Security

- **AES-256-GCM** — gradient payloads encrypted in transit
- **RSA-2048-OAEP-SHA256** — per-session AES key exchange
- **Supabase RLS** — row-level data isolation per user
- **3-tier RBAC** — `super_admin` > `admin` > `user`
- **Audit logs** — INSERT-only table, immutable trail
- **Rate limiting** — 5 login attempts / 15 min
- **HSTS + CSP + CORS** — enforced in production

## Super Admin

Account `sbhuvan847@gmail.com` is automatically promoted to `super_admin` via Supabase Edge Function on first login. Seed this via:

```sql
UPDATE auth.users SET raw_app_meta_data = raw_app_meta_data || '{"role":"super_admin"}'
WHERE email = 'sbhuvan847@gmail.com';
```

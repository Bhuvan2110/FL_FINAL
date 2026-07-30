"""
Auth API — Supabase Auth integration, RBAC, rate-limiting, audit logging.
"""
from fastapi import APIRouter, HTTPException, Request, Depends, status
from pydantic import BaseModel, EmailStr
import httpx
from app.core.config import get_settings
from app.core.security import get_server_public_key_pem
from app.api.dependencies import get_current_user, require_admin
from app.db.supabase_client import get_supabase

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()

# ── Rate Limiting (in-memory for local dev, Redis-backed in prod) ─────────────
_login_attempts: dict[str, list] = {}
MAX_ATTEMPTS = 5
WINDOW_SECONDS = 900  # 15 min


def _check_rate_limit(ip: str):
    import time
    now = time.time()
    attempts = _login_attempts.get(ip, [])
    attempts = [t for t in attempts if now - t < WINDOW_SECONDS]
    if len(attempts) >= MAX_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Try again in 15 minutes.",
        )
    attempts.append(now)
    _login_attempts[ip] = attempts


def _log_audit(user_id: str | None, action: str, resource: str, ip: str, detail: dict = None):
    try:
        sb = get_supabase()
        sb.table("audit_logs").insert({
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "ip": ip,
            "detail": detail or {},
        }).execute()
    except Exception:
        pass  # Audit log failure must not block the request


# ── Models ────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class SignupRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/login")
async def login(body: LoginRequest, request: Request):
    """Supabase Auth sign-in with rate limiting."""
    ip = request.client.host if request.client else "unknown"
    _check_rate_limit(ip)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.supabase_url}/auth/v1/token?grant_type=password",
            json={"email": body.email, "password": body.password},
            headers={"apikey": settings.supabase_anon_key, "Content-Type": "application/json"},
            timeout=15,
        )

    if resp.status_code != 200:
        _log_audit(None, "login_failed", "auth", ip, {"email": body.email})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    data = resp.json()
    _log_audit(data.get("user", {}).get("id"), "login_success", "auth", ip)
    return {
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token"),
        "user": {
            "id": data.get("user", {}).get("id"),
            "email": data.get("user", {}).get("email"),
            "role": data.get("user", {}).get("app_metadata", {}).get("role", "user"),
        },
    }


@router.post("/signup")
async def signup(body: SignupRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.supabase_url}/auth/v1/signup",
            json={"email": body.email, "password": body.password},
            headers={"apikey": settings.supabase_anon_key, "Content-Type": "application/json"},
            timeout=15,
        )
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Signup failed")
    data = resp.json()
    _log_audit(data.get("id"), "signup", "auth", ip)
    return {"message": "Check your email to confirm registration"}


@router.post("/refresh")
async def refresh_token(body: RefreshRequest):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.supabase_url}/auth/v1/token?grant_type=refresh_token",
            json={"refresh_token": body.refresh_token},
            headers={"apikey": settings.supabase_anon_key, "Content-Type": "application/json"},
            timeout=15,
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    data = resp.json()
    return {"access_token": data.get("access_token"), "refresh_token": data.get("refresh_token")}


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    return user


@router.get("/security/public-key")
async def get_public_key():
    """Server RSA public key for client-side AES key encryption."""
    return {"public_key": get_server_public_key_pem()}


@router.get("/audit-logs")
async def get_audit_logs(user: dict = Depends(require_admin)):
    sb = get_supabase()
    result = sb.table("audit_logs").select("*").order("timestamp", desc=True).limit(200).execute()
    return result.data

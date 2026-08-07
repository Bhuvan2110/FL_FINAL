"""Tests for API dependencies (JWT verification, role authorization)."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api.dependencies import get_current_user, require_admin, require_super_admin
from app.core.config import get_settings
from app.core.security import create_access_token


@pytest.mark.asyncio
async def test_get_current_user_local_jwt():
    settings = get_settings()
    token = create_access_token({"sub": "user123", "email": "user@example.com", "role": "user"}, settings.secret_key)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    user = await get_current_user(creds)
    assert user["id"] == "user123"
    assert user["email"] == "user@example.com"
    assert user["role"] == "user"


@pytest.mark.asyncio
async def test_get_current_user_guest_tokens():
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="guest-token")
    user = await get_current_user(creds)
    assert user["id"] == "guest"
    assert user["role"] == "guest"

    creds_g = HTTPAuthorizationCredentials(scheme="Bearer", credentials="mock-google-token")
    user_g = await get_current_user(creds_g)
    assert user_g["id"] == "mock-google-id"

    creds_e = HTTPAuthorizationCredentials(scheme="Bearer", credentials="mock-email-token")
    user_e = await get_current_user(creds_e)
    assert user_e["id"] == "mock-email-id"


@pytest.mark.asyncio
async def test_get_current_user_supabase_validation_success():
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid-supabase-token")
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "sb-user-123",
        "email": "sb@example.com",
        "user_metadata": {"full_name": "Supabase User"},
        "app_metadata": {"role": "user"},
    }

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get.return_value = mock_resp

    with patch("httpx.AsyncClient", return_value=mock_client):
        user = await get_current_user(creds)
        assert user["id"] == "sb-user-123"
        assert user["name"] == "Supabase User"


@pytest.mark.asyncio
async def test_get_current_user_invalid_token():
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid-token")
    
    mock_resp = MagicMock()
    mock_resp.status_code = 401

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get.return_value = mock_resp

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(creds)
        assert exc_info.value.status_code == 401


def test_require_admin():
    admin_user = {"id": "1", "role": "admin"}
    assert require_admin(admin_user) == admin_user

    regular_user = {"id": "2", "role": "user"}
    with pytest.raises(HTTPException) as exc:
        require_admin(regular_user)
    assert exc.value.status_code == 403


def test_require_super_admin():
    super_admin = {"id": "1", "role": "super_admin"}
    assert require_super_admin(super_admin) == super_admin

    admin_user = {"id": "2", "role": "admin"}
    with pytest.raises(HTTPException) as exc:
        require_super_admin(admin_user)
    assert exc.value.status_code == 403

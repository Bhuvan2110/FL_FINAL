"""Tests for Auth API endpoints."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture
def mock_supabase():
    """Mock Supabase client."""
    with patch('app.api.auth.get_supabase') as mock:
        yield mock


@pytest.fixture
def mock_httpx():
    """Mock httpx AsyncClient."""
    with patch('app.api.auth.httpx.AsyncClient') as mock:
        yield mock


def test_login_success(mock_httpx):
    """Test successful login."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "test_token",
        "refresh_token": "test_refresh",
        "user": {
            "id": "user123",
            "email": "test@example.com",
            "app_metadata": {"role": "admin"}
        }
    }
    
    mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
    
    response = client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "password123"
    })
    
    assert response.status_code == 200
    assert response.json()["access_token"] == "test_token"


def test_login_invalid_credentials(mock_httpx):
    """Test login with invalid credentials."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    
    mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
    
    response = client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "wrongpassword"
    })
    
    assert response.status_code == 401


def test_signup_success(mock_httpx):
    """Test successful signup."""
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"id": "user123"}
    
    mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
    
    response = client.post("/auth/signup", json={
        "email": "newuser@example.com",
        "password": "password123"
    })
    
    assert response.status_code == 200
    assert "Check your email" in response.json()["message"]


def test_signup_failure(mock_httpx):
    """Test signup failure."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    
    mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
    
    response = client.post("/auth/signup", json={
        "email": "newuser@example.com",
        "password": "password123"
    })
    
    assert response.status_code == 400


def test_refresh_token_success(mock_httpx):
    """Test successful token refresh."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "new_token",
        "refresh_token": "new_refresh"
    }
    
    mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
    
    response = client.post("/auth/refresh", json={"refresh_token": "old_refresh"})
    
    assert response.status_code == 200
    assert response.json()["access_token"] == "new_token"


def test_refresh_token_invalid(mock_httpx):
    """Test refresh with invalid token."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    
    mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
    
    response = client.post("/auth/refresh", json={"refresh_token": "invalid"})
    
    assert response.status_code == 401


def test_get_public_key():
    """Test getting public key endpoint."""
    response = client.get("/auth/security/public-key")
    
    assert response.status_code == 200
    assert "public_key" in response.json()


def test_rate_limiting():
    """Test rate limiting on login attempts."""
    from app.api import auth
    
    # Reset rate limiting
    auth._login_attempts.clear()
    
    # Make multiple requests from same IP
    mock_response = AsyncMock()
    mock_response.status_code = 401
    
    with patch('app.api.auth.httpx.AsyncClient') as mock_httpx:
        mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        for i in range(5):
            response = client.post("/auth/login", json={
                "email": f"test{i}@example.com",
                "password": "wrong"
            })
            assert response.status_code == 401
        
        # 6th attempt should be rate limited
        response = client.post("/auth/login", json={
            "email": "test@example.com",
            "password": "wrong"
        })
        assert response.status_code == 429


def test_audit_logging(mock_supabase):
    """Test audit logging functionality."""
    from app.api.auth import _log_audit
    
    mock_sb = MagicMock()
    mock_supabase.return_value = mock_sb
    
    _log_audit("user123", "login_success", "auth", "127.0.0.1")
    
    mock_sb.table.assert_called_with("audit_logs")


def test_audit_logging_failure():
    """Test that audit logging failure doesn't break flow."""
    from app.api.auth import _log_audit
    
    with patch('app.api.auth.get_supabase', side_effect=Exception("DB error")):
        # Should not raise
        _log_audit("user123", "login_success", "auth", "127.0.0.1")


def test_check_rate_limit_under_threshold():
    """Test that rate limiting allows requests under threshold."""
    # Reset
    from app.api import auth
    from app.api.auth import _check_rate_limit
    auth._login_attempts.clear()
    
    # Should not raise
    for i in range(3):
        _check_rate_limit("192.168.1.1")


def test_invalid_email_format():
    """Test login with invalid email format."""
    response = client.post("/auth/login", json={
        "email": "not-an-email",
        "password": "password123"
    })
    
    # Pydantic validation should catch this
    assert response.status_code == 422

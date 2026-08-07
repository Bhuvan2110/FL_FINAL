"""Pytest configuration and shared fixtures."""
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(autouse=True)
def reset_imports():
    """Reset imports between tests to avoid state pollution."""
    yield
    # Cleanup after each test
    if 'app.core.config' in sys.modules:
        # Reset settings cache
        from app.core import config
        config.get_settings.cache_clear()


@pytest.fixture
def mock_supabase():
    """Mock Supabase client."""
    with patch('app.db.supabase_client.get_supabase') as mock:
        yield mock


@pytest.fixture
def mock_settings():
    """Mock settings."""
    settings = MagicMock()
    settings.supabase_url = "https://test.supabase.co"
    settings.supabase_anon_key = "test-key"
    settings.redis_url = "redis://localhost:6379/0"
    settings.cors_origins = ["http://localhost:3000"]
    settings.environment = "test"
    settings.mlflow_tracking_uri = "http://localhost:5000"
    
    with patch('app.core.config.get_settings', return_value=settings):
        yield settings


@pytest.fixture
def test_user():
    """Create test user."""
    return {
        "id": "test-user-id",
        "email": "test@example.com",
        "role": "user"
    }


@pytest.fixture
def test_admin():
    """Create test admin user."""
    return {
        "id": "test-admin-id",
        "email": "admin@example.com",
        "role": "admin"
    }

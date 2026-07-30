"""Tests for application configuration."""
from unittest.mock import patch
from app.core.config import Settings, get_settings


def test_settings_defaults():
    """Test that settings have proper defaults."""
    settings = Settings(_env_file=None)
    
    assert "supabase.co" in settings.supabase_url
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.environment == "development"


def test_cors_origins_parsing():
    """Test CORS origins parsing from comma-separated string."""
    settings = Settings(cors_origins_raw="http://localhost:3000, http://localhost:5173")
    
    origins = settings.cors_origins
    assert len(origins) == 2
    assert "http://localhost:3000" in origins
    assert "http://localhost:5173" in origins


def test_cors_origins_with_whitespace():
    """Test CORS origins parsing handles whitespace."""
    settings = Settings(cors_origins_raw="  http://example.com  ,  http://test.com  ")
    
    origins = settings.cors_origins
    assert "http://example.com" in origins
    assert "http://test.com" in origins
    assert len(origins) == 2


def test_cors_origins_empty_string():
    """Test CORS origins with empty string."""
    settings = Settings(cors_origins_raw="")
    
    origins = settings.cors_origins
    assert origins == []


def test_cors_origins_single_entry():
    """Test CORS origins with single entry."""
    settings = Settings(cors_origins_raw="http://localhost:3000")
    
    origins = settings.cors_origins
    assert origins == ["http://localhost:3000"]


def test_get_settings_caching():
    """Test that get_settings returns cached instance."""
    settings1 = get_settings()
    settings2 = get_settings()
    
    assert settings1 is settings2


def test_settings_with_env_vars():
    """Test settings picks up from environment."""
    with patch.dict('os.environ', {
        'SUPABASE_URL': 'https://custom.supabase.co',
        'SUPABASE_ANON_KEY': 'custom-key',
        'REDIS_URL': 'redis://custom:6379/0',
        'ENVIRONMENT': 'production'
    }):
        # Clear cache first
        get_settings.cache_clear()
        
        settings = Settings()
        
        assert settings.supabase_url == 'https://custom.supabase.co'
        assert settings.supabase_anon_key == 'custom-key'
        assert settings.redis_url == 'redis://custom:6379/0'
        assert settings.environment == 'production'


def test_settings_mlflow_uri():
    """Test MLflow tracking URI configuration."""
    settings = Settings(mlflow_tracking_uri="http://mlflow.example.com:5000")
    
    assert settings.mlflow_tracking_uri == "http://mlflow.example.com:5000"


def test_settings_secret_key():
    """Test secret key configuration."""
    settings = Settings(secret_key="my-super-secret-key")
    
    assert settings.secret_key == "my-super-secret-key"


def test_cors_origins_with_protocols():
    """Test CORS origins with different protocols."""
    settings = Settings(cors_origins_raw="http://localhost:3000, https://api.example.com, ws://socket.example.com")
    
    origins = settings.cors_origins
    assert len(origins) == 3
    assert any("http://" in o for o in origins)
    assert any("https://" in o for o in origins)
    assert any("ws://" in o for o in origins)


def test_cors_origins_filtering_empty():
    """Test that empty entries in CORS are filtered."""
    settings = Settings(cors_origins_raw="http://localhost:3000,  , http://example.com")
    
    origins = settings.cors_origins
    # Empty string after comma should be filtered
    assert len(origins) == 2

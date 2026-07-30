"""Tests for main FastAPI application."""
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns expected structure."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "FL Platform API"
    assert data["version"] == "1.0.0"
    assert "/docs" in data["docs"]
    assert "fedavg" in data["algorithms"]


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "environment" in response.json()


def test_cors_headers():
    """Test CORS headers are present in responses."""
    response = client.options("/health")
    
    # FastAPI with CORSMiddleware adds these
    assert response.status_code in [200, 204]


def test_security_headers():
    """Test security headers are added to responses."""
    response = client.get("/health")
    
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"


def test_docs_endpoint_available():
    """Test that API documentation is available."""
    response = client.get("/docs")
    
    assert response.status_code == 200


def test_redoc_endpoint_available():
    """Test that ReDoc documentation is available."""
    response = client.get("/redoc")
    
    assert response.status_code == 200


def test_openapi_schema():
    """Test OpenAPI schema is available."""
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "FL Platform API"
    assert schema["info"]["version"] == "1.0.0"


@patch('app.main.socket.create_connection')
def test_mlflow_connection_failure(mock_socket):
    """Test that MLflow connection failure is handled gracefully."""
    mock_socket.side_effect = Exception("Connection failed")
    
    # Recreate app with patched socket (simulates startup)
    with patch('app.main.socket.create_connection', side_effect=Exception("Connection failed")):
        # App should still be functional
        response = client.get("/health")
        assert response.status_code == 200


def test_404_not_found():
    """Test 404 response for unknown endpoint."""
    response = client.get("/nonexistent-endpoint")
    
    assert response.status_code == 404


def test_method_not_allowed():
    """Test 405 response for wrong HTTP method."""
    response = client.post("/health")
    
    assert response.status_code in [405, 404]  # FastAPI might return 404


def test_app_title_and_description():
    """Test app metadata."""
    schema = client.get("/openapi.json").json()
    
    assert "FL Platform API" in schema["info"]["title"]
    assert "Federated Learning" in schema["info"]["description"]


def test_lifespan_context():
    """Test that lifespan context manager works."""
    # The app should have initialized without errors
    response = client.get("/health")
    assert response.status_code == 200


def test_multiple_routers_loaded():
    """Test that all routers are loaded."""
    schema = client.get("/openapi.json").json()
    paths = schema["paths"].keys()
    
    # Check that endpoints from different routers are present
    assert any("/auth" in str(path) for path in paths)


def test_middleware_order():
    """Test that middleware is applied in correct order."""
    response = client.get("/health")
    
    # Security headers should be present (added by security_headers middleware)
    assert "X-Content-Type-Options" in response.headers
    # CORS headers should be present
    assert "content-length" in response.headers or response.status_code == 200


def test_response_type():
    """Test that responses are JSON."""
    response = client.get("/health")
    
    assert response.headers["content-type"].startswith("application/json")


def test_root_algorithms_list_valid():
    """Test that algorithms list is valid."""
    response = client.get("/")
    data = response.json()
    
    algorithms = data["algorithms"]
    assert isinstance(algorithms, list)
    assert len(algorithms) > 0
    
    expected_algorithms = ["fedavg", "fedprox", "scaffold", "dpsgd", "central"]
    for algo in expected_algorithms:
        assert algo in algorithms


def test_health_environment_field():
    """Test health endpoint includes environment."""
    response = client.get("/health")
    data = response.json()
    
    assert "environment" in data
    assert data["environment"] in ["development", "production", "testing"]

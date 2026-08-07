"""Tests for Prediction API endpoints."""
import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.main import app

client = TestClient(app)


def test_predict_single_model_not_found():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1"}
    mock_sb = MagicMock()
    single_mock = MagicMock()
    single_mock.execute.return_value.data = None
    mock_sb.table().select().eq().single.return_value = single_mock

    with patch("app.api.predict.get_supabase", return_value=mock_sb):
        response = client.post("/predict/single", json={"model_id": "m-missing", "features": {"f1": 1.0}})
        assert response.status_code == 404
        assert response.json()["detail"] == "Model not found"

    app.dependency_overrides.clear()


def test_predict_single_success():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1"}
    mock_sb = MagicMock()
    
    single_mock = MagicMock()
    single_mock.execute.return_value.data = {
        "id": "model-1",
        "weights_path": "models/weights.json"
    }
    mock_sb.table().select().eq().single.return_value = single_mock

    model_json = json.dumps({
        "weights": [0.5, -0.2],
        "bias": 0.1,
        "scalers": [{"min": 0.0, "range": 10.0}, {"min": 0.0, "range": 100.0}],
        "feature_names": ["f1", "f2"]
    }).encode("utf-8")

    mock_sb.storage.from_().download.return_value = model_json

    insert_mock = MagicMock()
    insert_mock.execute.return_value.data = [{"id": "pred-1"}]
    mock_sb.table().insert.return_value = insert_mock

    with patch("app.api.predict.get_supabase", return_value=mock_sb):
        response = client.post("/predict/single", json={"model_id": "model-1", "features": {"f1": 5.0, "f2": 50.0}})
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "class_label" in data

    app.dependency_overrides.clear()


def test_prediction_history():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1"}
    mock_sb = MagicMock()
    limit_mock = MagicMock()
    limit_mock.execute.return_value.data = [
        {"id": "p1", "output": 1, "confidence": 0.95}
    ]
    mock_sb.table().select().eq().order().limit.return_value = limit_mock

    with patch("app.api.predict.get_supabase", return_value=mock_sb):
        response = client.get("/predict/history")
        assert response.status_code == 200
        assert len(response.json()) == 1

    app.dependency_overrides.clear()


def test_list_available_models():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1", "role": "user"}
    mock_sb = MagicMock()

    # Query 1: experiments
    exp_exec = MagicMock()
    exp_exec.execute.return_value.data = [
        {"id": "exp-1", "algorithm": "fedavg", "created_at": "2026-01-01T00:00:00Z"}
    ]
    
    # Query 2: models for exp-1
    mod_exec = MagicMock()
    mod_exec.execute.return_value.data = [
        {"id": "mod-1", "experiment_id": "exp-1", "weights_path": "path"}
    ]

    mock_sb.table().select().eq().eq.return_value = exp_exec
    mock_sb.table().select().eq.side_effect = [exp_exec, mod_exec]

    with patch("app.api.predict.get_supabase", return_value=mock_sb):
        response = client.get("/predict/models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)

    app.dependency_overrides.clear()

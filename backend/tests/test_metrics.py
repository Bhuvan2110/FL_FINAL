"""Tests for Metrics API endpoints."""
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.main import app

client = TestClient(app)


def test_compare_metrics():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1", "role": "user"}
    mock_sb = MagicMock()

    exps_mock = MagicMock()
    exps_mock.execute.return_value.data = [
        {"id": "exp-1", "algorithm": "fedavg", "status": "completed"}
    ]
    mock_sb.table().select().eq().eq.return_value = exps_mock

    metrics_mock = MagicMock()
    metrics_mock.execute.return_value.data = {
        "accuracy": 0.92,
        "f1": 0.90,
        "auc": 0.95
    }
    pb_mock = MagicMock()
    pb_mock.execute.return_value.data = [
        {"epsilon": 2.5}
    ]

    mock_sb.table().select().eq().single.return_value = metrics_mock
    mock_sb.table().select().eq().order().limit.return_value = pb_mock

    with patch("app.api.metrics.get_supabase", return_value=mock_sb):
        response = client.get("/metrics/compare?models=fedavg")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["algorithm"] == "fedavg"
        assert data[0]["accuracy"] == 0.92

    app.dependency_overrides.clear()


def test_privacy_utility_curve():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1", "role": "user"}
    mock_sb = MagicMock()

    exps_mock = MagicMock()
    exps_mock.execute.return_value.data = [
        {"id": "exp-dp-1"}
    ]
    mock_sb.table().select().eq().eq().eq.return_value = exps_mock

    rounds_mock = MagicMock()
    rounds_mock.execute.return_value.data = [{"round_num": 1, "accuracy": 0.80, "val_accuracy": 0.82}]
    
    pb_mock = MagicMock()
    pb_mock.execute.return_value.data = [{"round_num": 1, "epsilon": 0.5}]

    mock_sb.table().select().eq().order.side_effect = [rounds_mock, pb_mock]

    with patch("app.api.metrics.get_supabase", return_value=mock_sb):
        response = client.get("/metrics/privacy-utility")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["epsilon"] == 0.5

    app.dependency_overrides.clear()


def test_get_experiment_metrics():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1"}
    mock_sb = MagicMock()

    metrics_mock = MagicMock()
    metrics_mock.execute.return_value.data = {"accuracy": 0.88}
    mock_sb.table().select().eq().single.return_value = metrics_mock

    rounds_mock = MagicMock()
    rounds_mock.execute.return_value.data = [{"round_num": 1, "loss": 0.3}]
    
    pb_mock = MagicMock()
    pb_mock.execute.return_value.data = [{"round_num": 1, "epsilon": 1.0}]

    mock_sb.table().select().eq().order.side_effect = [rounds_mock, pb_mock]

    with patch("app.api.metrics.get_supabase", return_value=mock_sb):
        response = client.get("/metrics/experiment/exp-123")
        assert response.status_code == 200
        data = response.json()
        assert data["metrics"]["accuracy"] == 0.88
        assert len(data["rounds"]) == 1

    app.dependency_overrides.clear()

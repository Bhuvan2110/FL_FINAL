"""Tests for Training API endpoints."""
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.main import app

client = TestClient(app)


def test_start_training_invalid_algorithm():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1"}
    response = client.post("/training/start", json={"dataset_id": "ds-1", "algorithm": "invalid_alg"})
    assert response.status_code == 400
    assert "Algorithm must be one of" in response.json()["detail"]
    app.dependency_overrides.clear()


def test_start_training_dataset_not_found():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1"}
    mock_sb = MagicMock()
    exec_mock = MagicMock()
    exec_mock.execute.return_value.data = None
    mock_sb.table().select().eq().eq.return_value = exec_mock

    with patch("app.api.training.get_supabase", return_value=mock_sb):
        response = client.post("/training/start", json={"dataset_id": "ds-missing", "algorithm": "fedavg"})
        assert response.status_code == 404
        assert response.json()["detail"] == "Dataset not found"

    app.dependency_overrides.clear()


def test_start_training_success_background_fallback():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1"}
    
    mock_sb = MagicMock()
    ds_mock = MagicMock()
    ds_mock.execute.return_value.data = [{"id": "ds-1"}]
    mock_sb.table().select().eq().eq.return_value = ds_mock

    single_ds_mock = MagicMock()
    single_ds_mock.execute.return_value.data = {"id": "ds-1", "storage_path": "datasets/user1/test.csv"}
    mock_sb.table().select().eq().single.return_value = single_ds_mock

    ins_mock = MagicMock()
    ins_mock.execute.return_value.data = [{"id": "exp-new-123"}]
    mock_sb.table().insert.return_value = ins_mock

    upd_mock = MagicMock()
    upd_mock.execute.return_value.data = [{"id": "exp-new-123"}]
    mock_sb.table().update().eq.return_value = upd_mock

    csv_bytes = b"Age,Sex,Target\n20,M,0\n30,F,1\n25,M,0\n35,F,1\n40,M,0\n45,F,1\n50,M,0\n55,F,1\n60,M,0\n65,F,1\n"
    mock_sb.storage.from_().download.return_value = csv_bytes

    with (
        patch("app.api.training.get_supabase", return_value=mock_sb),
        patch("app.db.supabase_client.get_supabase", return_value=mock_sb),
        patch("socket.create_connection", side_effect=OSError("No connection")),
    ):
        response = client.post("/training/start", json={"dataset_id": "ds-1", "algorithm": "fedavg", "n_rounds": 2})
        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == "exp-new-123"
        assert "task_id" in data

    app.dependency_overrides.clear()


def test_list_experiments():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1"}
    mock_sb = MagicMock()
    exec_mock = MagicMock()
    exec_mock.execute.return_value.data = [
        {"id": "exp-1", "algorithm": "fedavg", "status": "completed"}
    ]
    mock_sb.table().select().eq().order.return_value = exec_mock

    with patch("app.api.training.get_supabase", return_value=mock_sb):
        response = client.get("/training/list")
        assert response.status_code == 200
        assert len(response.json()) == 1

    app.dependency_overrides.clear()


def test_compare_experiments_training():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1", "role": "user"}
    mock_sb = MagicMock()

    exps_mock = MagicMock()
    exps_mock.execute.return_value.data = [
        {"id": "exp-1", "algorithm": "fedavg", "created_at": "2026-01-01T00:00:00Z"}
    ]
    mock_sb.table().select().eq().eq.return_value = exps_mock

    metrics_mock = MagicMock()
    metrics_mock.execute.return_value.data = {"accuracy": 0.90}
    mock_sb.table().select().eq().single.return_value = metrics_mock

    rounds_mock = MagicMock()
    rounds_mock.execute.return_value.data = [{"round_num": 1, "loss": 0.5}]

    pb_mock = MagicMock()
    pb_mock.execute.return_value.data = [{"round_num": 1, "epsilon": 0.1}]

    mock_sb.table().select().eq().order.side_effect = [rounds_mock, pb_mock]

    with patch("app.api.training.get_supabase", return_value=mock_sb):
        response = client.get("/training/compare")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["metrics"]["accuracy"] == 0.90

    app.dependency_overrides.clear()


def test_get_training_status_not_found():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1"}
    mock_sb = MagicMock()
    single_mock = MagicMock()
    single_mock.execute.return_value.data = None
    mock_sb.table().select().eq().single.return_value = single_mock

    with patch("app.api.training.get_supabase", return_value=mock_sb):
        response = client.get("/training/exp-missing/status")
        assert response.status_code == 404

    app.dependency_overrides.clear()


def test_get_training_status_completed():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1"}
    mock_sb = MagicMock()

    single_exp_mock = MagicMock()
    single_exp_mock.execute.return_value.data = {"id": "exp-1", "status": "completed"}
    
    single_metrics_mock = MagicMock()
    single_metrics_mock.execute.return_value.data = {"accuracy": 0.92}

    mock_sb.table().select().eq().single.side_effect = [single_exp_mock, single_metrics_mock]

    rounds_mock = MagicMock()
    rounds_mock.execute.return_value.data = [{"round_num": 1, "loss": 0.2}]
    
    pb_mock = MagicMock()
    pb_mock.execute.return_value.data = [{"round_num": 1, "epsilon": 1.0}]

    mock_sb.table().select().eq().order.side_effect = [rounds_mock, pb_mock]

    with patch("app.api.training.get_supabase", return_value=mock_sb):
        response = client.get("/training/exp-1/status")
        assert response.status_code == 200
        data = response.json()
        assert data["experiment"]["status"] == "completed"
        assert data["metrics"]["accuracy"] == 0.92

    app.dependency_overrides.clear()


def test_delete_experiment_not_found():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1"}
    mock_sb = MagicMock()
    single_mock = MagicMock()
    single_mock.execute.return_value.data = None
    mock_sb.table().select().eq().single.return_value = single_mock

    with patch("app.api.training.get_supabase", return_value=mock_sb):
        response = client.delete("/training/exp-missing")
        assert response.status_code == 404

    app.dependency_overrides.clear()


def test_delete_experiment_unauthorized():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1", "role": "user"}
    mock_sb = MagicMock()
    single_mock = MagicMock()
    single_mock.execute.return_value.data = {"id": "exp-1", "user_id": "other_user"}
    mock_sb.table().select().eq().single.return_value = single_mock

    with patch("app.api.training.get_supabase", return_value=mock_sb):
        response = client.delete("/training/exp-1")
        assert response.status_code == 403

    app.dependency_overrides.clear()


def test_delete_experiment_success():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1", "role": "user"}
    mock_sb = MagicMock()
    single_mock = MagicMock()
    single_mock.execute.return_value.data = {"id": "exp-1", "user_id": "user1"}
    mock_sb.table().select().eq().single.return_value = single_mock

    with patch("app.api.training.get_supabase", return_value=mock_sb):
        response = client.delete("/training/exp-1")
        assert response.status_code == 200
        assert response.json()["message"] == "Experiment deleted"

    app.dependency_overrides.clear()

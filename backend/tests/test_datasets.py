"""Tests for Datasets API endpoints."""
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.main import app

client = TestClient(app)


def test_upload_dataset_non_csv():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1", "email": "user@test.com"}
    response = client.post("/dataset/upload", files={"file": ("test.txt", b"hello world", "text/plain")})
    assert response.status_code == 400
    assert "Only CSV files are accepted" in response.json()["detail"]
    app.dependency_overrides.clear()


def test_upload_dataset_insufficient_rows():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1", "email": "user@test.com"}
    short_csv = b"col1,col2\n1,2\n3,4\n"
    response = client.post("/dataset/upload", files={"file": ("short.csv", short_csv, "text/csv")})
    assert response.status_code == 400
    assert "at least 10 rows" in response.json()["detail"]
    app.dependency_overrides.clear()


def test_upload_dataset_success():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1", "name": "Test User", "email": "user@test.com"}
    
    rows = ["Age,Feature1,Target"] + [f"{20+i},{i*1.5},{i%2}" for i in range(15)]
    valid_csv = "\n".join(rows).encode("utf-8")

    mock_sb = MagicMock()
    mock_sb.storage.from_().upload.return_value = True
    insert_mock = MagicMock()
    insert_mock.execute.return_value.data = [{"id": "ds-123"}]
    mock_sb.table().insert.return_value = insert_mock

    with patch("app.api.datasets.get_supabase", return_value=mock_sb):
        response = client.post("/dataset/upload", files={"file": ("dataset.csv", valid_csv, "text/csv")})
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "ds-123"
        assert data["filename"] == "dataset.csv"
        assert data["row_count"] == 15

    app.dependency_overrides.clear()


def test_preview_dataset_not_found():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1", "email": "user@test.com"}
    mock_sb = MagicMock()
    single_mock = MagicMock()
    single_mock.execute.return_value.data = None
    mock_sb.table().select().eq().eq().single.return_value = single_mock

    with patch("app.api.datasets.get_supabase", return_value=mock_sb):
        response = client.get("/dataset/preview/ds-missing")
        assert response.status_code == 404

    app.dependency_overrides.clear()


def test_preview_dataset_success():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1", "email": "user@test.com"}
    mock_sb = MagicMock()
    
    single_mock = MagicMock()
    single_mock.execute.return_value.data = {
        "id": "ds-123",
        "storage_path": "datasets/user1/dataset.csv",
    }
    mock_sb.table().select().eq().eq().single.return_value = single_mock

    csv_data = b"sl.no,Age,Target\n1,25,0\n2,30,1\n"
    mock_sb.storage.from_().download.return_value = csv_data

    with patch("app.api.datasets.get_supabase", return_value=mock_sb):
        response = client.get("/dataset/preview/ds-123")
        assert response.status_code == 200
        data = response.json()
        assert data["total_rows"] == 2
        assert "headers" in data
        assert "rows" in data

    app.dependency_overrides.clear()


def test_list_datasets():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1", "email": "user@test.com"}
    mock_sb = MagicMock()
    exec_mock = MagicMock()
    exec_mock.execute.return_value.data = [
        {"id": "ds-1", "filename": "data1.csv", "row_count": 100}
    ]
    mock_sb.table().select().eq().order.return_value = exec_mock

    with patch("app.api.datasets.get_supabase", return_value=mock_sb):
        response = client.get("/dataset/list")
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["filename"] == "data1.csv"

    app.dependency_overrides.clear()


def test_delete_dataset():
    app.dependency_overrides[get_current_user] = lambda: {"id": "user1", "email": "user@test.com"}
    mock_sb = MagicMock()
    single_mock = MagicMock()
    single_mock.execute.return_value.data = {
        "id": "ds-1",
        "storage_path": "path/ds-1.csv",
    }
    mock_sb.table().select().eq().eq().single.return_value = single_mock

    with patch("app.api.datasets.get_supabase", return_value=mock_sb):
        response = client.delete("/dataset/ds-1")
        assert response.status_code == 200
        assert response.json()["message"] == "Dataset deleted"

    app.dependency_overrides.clear()

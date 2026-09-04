from pathlib import Path
import sys

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app


client = TestClient(app)


VALID_PAYLOAD = {
    "age": 35,
    "job": "technician",
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "balance": 1500.0,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day_of_week": 15,
    "month": "may",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data


def test_predict_valid_input():
    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert "model_version" in data

    assert data["prediction"] in [0, 1]

    if data["probability"] is not None:
        assert 0.0 <= data["probability"] <= 1.0


def test_predict_invalid_input():
    invalid_payload = VALID_PAYLOAD.copy()

    invalid_payload["age"] = 10

    response = client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422
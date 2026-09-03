from fastapi.testclient import (
    TestClient
)

from src.api.main import app


client = TestClient(
    app
)


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["model_loaded"] is True


def test_prediction():

    payload = {

        "age": 35,

        "job": "management",

        "marital": "married",

        "education": "tertiary",

        "default": "no",

        "balance": 1500,

        "housing": "yes",

        "loan": "no",

        "contact": "cellular",

        "day_of_week": 15,

        "month": "may",

        "campaign": 1,

        "pdays": -1,

        "previous": 0,

        "poutcome": "unknown"
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert "probability" in data

    assert "model_version" in data

    assert data["prediction"] in [
        0,
        1
    ]

    assert 0 <= data[
        "probability"
    ] <= 1


def test_invalid_age():

    payload = {

        "age": 150,

        "job": "management",

        "marital": "married",

        "education": "tertiary",

        "default": "no",

        "balance": 1500,

        "housing": "yes",

        "loan": "no",

        "contact": "cellular",

        "day_of_week": 15,

        "month": "may",

        "campaign": 1,

        "pdays": -1,

        "previous": 0,

        "poutcome": "unknown"
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422
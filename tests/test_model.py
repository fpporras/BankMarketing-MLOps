from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import (
    build_features
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_model.joblib"
)


def test_model_exists():

    assert MODEL_PATH.exists()


def test_model_prediction():

    model = joblib.load(
        MODEL_PATH
    )

    sample = pd.DataFrame([{

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
    }])

    X = build_features(
        sample
    )

    prediction = model.predict(
        X
    )[0]

    probability = model.predict_proba(
        X
    )[0][1]

    assert prediction in [0, 1]

    assert 0 <= probability <= 1
from pathlib import Path
import sys

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import build_features


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank_marketing.csv"


def test_model_exists():
    assert MODEL_PATH.exists(), f"No se encontró el modelo en: {MODEL_PATH}"


def test_model_generates_valid_prediction():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    df_features = build_features(df)

    X = df_features.drop(columns=["y"])

    sample = X.iloc[[0]]

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_model_generates_valid_probability():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    df_features = build_features(df)

    X = df_features.drop(columns=["y"])

    sample = X.iloc[[0]]

    probability = model.predict_proba(sample)[0][1]

    assert 0.0 <= probability <= 1.0
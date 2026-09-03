from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_model.joblib"
)


def evaluate_production_batch(
    batch_file
):

    model = joblib.load(
        MODEL_PATH
    )

    df = pd.read_csv(
        batch_file
    )

    X = df.drop(
        columns=["y"]
    )

    y = (
        df["y"]
        .map({
            "no": 0,
            "yes": 1
        })
    )

    predictions = model.predict(
        X
    )

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    metrics = {

        "precision":
            precision_score(
                y,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y,
                predictions,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y,
                probabilities
            )
    }

    return metrics
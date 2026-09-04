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
    """
    Evalúa el modelo actual contra un batch
    de producción que contiene el target real.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"No se encontró el modelo en: "
            f"{MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    df = pd.read_csv(
        batch_file
    )

    if "y" not in df.columns:

        raise ValueError(
            "El batch de producción debe "
            "contener la columna 'y'."
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

    if y.isna().any():

        raise ValueError(
            "La columna 'y' contiene "
            "valores no válidos."
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
            float(
                precision_score(
                    y,
                    predictions,
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    y,
                    predictions,
                    zero_division=0
                )
            ),

        "f1":
            float(
                f1_score(
                    y,
                    predictions,
                    zero_division=0
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y,
                    probabilities
                )
            )
    }

    return metrics


def monitor_model(
    batch_file
):
    """
    Ejecuta model monitoring sobre un batch
    y devuelve un DataFrame.
    """

    metrics = evaluate_production_batch(
        batch_file
    )

    return pd.DataFrame(
        [metrics]
    )
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features.build_features import build_features


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_model.joblib"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bank_marketing.csv"
)

TARGET_COLUMN = "y"
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# CÁLCULO DE MÉTRICAS
# ============================================================

def calculate_model_metrics(
    y_true,
    y_pred,
    y_proba
):
    """
    Calcula las métricas requeridas para O3:
    Precision, Recall, F1 y AUC.
    """

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        y_proba
    )

    return pd.DataFrame([
        {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4),
        }
    ])


# ============================================================
# MONITOREO DEL MODELO
# ============================================================

def monitor_model():
    """
    Carga el modelo real y calcula las métricas de O3
    sobre el conjunto de test reproducido con la misma
    configuración utilizada durante entrenamiento.
    """

    # Verificar modelo
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en:\n{MODEL_PATH}"
        )

    # Verificar dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en:\n{DATA_PATH}"
        )

    # Cargar datos
    df = pd.read_csv(DATA_PATH)

    # Aplicar el mismo Feature Engineering
    # utilizado antes del entrenamiento
    df_features = build_features(df)

    # Separar X / y
    X = df_features.drop(
        columns=[TARGET_COLUMN]
    )

    y = (
        df_features[TARGET_COLUMN]
        .map({
            "no": 0,
            "yes": 1
        })
        .astype(int)
    )

    # Reproducir exactamente el split del entrenamiento
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Cargar pipeline completo
    model = joblib.load(MODEL_PATH)

    # Predicción de clases
    y_pred = model.predict(X_test)

    # Probabilidad de la clase positiva
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calcular métricas
    report = calculate_model_metrics(
        y_test,
        y_pred,
        y_proba
    )

    return report


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    report = monitor_model()

    print("\nO3 - MODEL MONITORING")
    print(report.to_string(index=False))
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

from src.api.schemas import BankMarketingRequest, PredictionResponse
from src.features.build_features import build_features


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

MODEL_VERSION = "1.0.0"


# ============================================================
# CARGAR MODELO
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Bank Marketing ML API",
    description=(
        "API de inferencia para predecir la suscripción "
        "a un depósito bancario."
    ),
    version=MODEL_VERSION,
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
    }


# ============================================================
# PREDICT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: BankMarketingRequest):

    try:

        # ----------------------------------------------------
        # 1. Convertir request a DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [data.model_dump()]
        )

        # ----------------------------------------------------
        # 2. Aplicar EXACTAMENTE el mismo Feature Engineering
        #    utilizado durante entrenamiento
        # ----------------------------------------------------

        input_features = build_features(
            input_data
        )

        # ----------------------------------------------------
        # 3. Separar target si existiera
        # ----------------------------------------------------

        if "y" in input_features.columns:

            input_features = input_features.drop(
                columns=["y"]
            )

        # ----------------------------------------------------
        # 4. Realizar predicción
        # ----------------------------------------------------

        prediction = model.predict(
            input_features
        )[0]

        # ----------------------------------------------------
        # 5. Probabilidad
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(
                input_features
            )[0][1]

        else:

            probability = None

        # ----------------------------------------------------
        # 6. Respuesta
        # ----------------------------------------------------

        return PredictionResponse(
            prediction=int(prediction),
            probability=(
                float(probability)
                if probability is not None
                else None
            ),
            model_version=MODEL_VERSION,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
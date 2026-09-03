import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import BankMarketingRequest, PredictionResponse
from src.features.build_features import build_features
from src.monitoring.system_monitor import (
    get_system_metrics,
    record_request,
)

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
    raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

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
# METRICS ENDPOINT
# ============================================================

@app.get("/metrics")
def metrics():
    return get_system_metrics()


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict", response_model=PredictionResponse)
def predict(data: BankMarketingRequest):
    # Registrar el tiempo de inicio para calcular la latencia
    start_time = time.perf_counter()

    try:
        # 1. Convertir request a DataFrame
        input_data = pd.DataFrame([data.model_dump()])

        # 2. Aplicar Feature Engineering
        input_features = build_features(input_data)

        # 3. Separar target si existiera
        if "y" in input_features.columns:
            input_features = input_features.drop(columns=["y"])

        # 4. Realizar predicción
        prediction = model.predict(input_features)[0]

        # 5. Probabilidad
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_features)[0][1]
        else:
            probability = None

        # 6. Calcular latencia e informar petición exitosa
        latency_ms = (time.perf_counter() - start_time) * 1000
        record_request(latency_ms=latency_ms, error=False)

        # 7. Respuesta
        return PredictionResponse(
            prediction=int(prediction),
            probability=(
                float(probability) if probability is not None else None
            ),
            model_version=MODEL_VERSION,
        )

    except Exception as e:
        # Registrar latencia en caso de fallo antes de lanzar el error
        latency_ms = (time.perf_counter() - start_time) * 1000
        record_request(latency_ms=latency_ms, error=True)

        raise HTTPException(status_code=500, detail=str(e))
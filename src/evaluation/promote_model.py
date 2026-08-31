import mlflow
from mlflow.tracking import MlflowClient

# ============================================================
# CONFIGURACIÓN DE UMBRALES Y REGISTRY
# ============================================================

# Umbrales para desarrollo
MIN_F1 = 0.40      
MIN_RECALL = 0.55   
MIN_ROC_AUC = 0.70 

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "bank-marketing-classifier"

# ============================================================
# VALIDACIÓN DE MÉTRICAS
# ============================================================
def validate_model(metrics: dict) -> bool:
    """Verifica si el modelo cumple con los umbrales mínimos de negocio."""
    checks = {
        "f1": metrics.get("f1", 0) >= MIN_F1,
        "recall": metrics.get("recall", 0) >= MIN_RECALL,
        "roc_auc": metrics.get("roc_auc", 0) >= MIN_ROC_AUC,
    }

    print("\n" + "=" * 45)
    print("MODEL VALIDATION FOR PROMOTION")
    print("=" * 45)

    for metric, passed in checks.items():
        val = metrics.get(metric, 0.0)
        status = "PASS" if passed else "FAIL"
        print(f"{metric.upper():<8}: {val:.4f} -> {status}")

    return all(checks.values())

# ============================================================
# PROMOCIÓN EN MLFLOW MODEL REGISTRY
# ============================================================
def promote_registered_model(model_name: str, metrics: dict):
    """
    Si valida las métricas, le asigna el alias 'champion' en MLflow.
    """
    if not validate_model(metrics):
        print("\nEl modelo NO cumplió los umbrales requeridos. No será promocionado.")
        return False

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    # Obtener la última versión del modelo registrado
    latest_versions = client.get_latest_versions(name=model_name)
    if not latest_versions:
        print(f"No se encontraron versiones registradas para '{model_name}'.")
        return False
        
    latest_version = latest_versions[-1].version

    # Asignar alias 'champion' a la versión aprobada (MLflow v2+)
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=latest_version
    )

    print("\n" + "=" * 45)
    print(f"ÉXITO: Modelo '{model_name}' versión {latest_version} promocionado a 'champion'.")
    print("=" * 45)
    return True

# ============================================================
# EJECUCIÓN AUTÓNOMA O IMPORTADA
# ============================================================
if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # Ejemplo: Obtener métricas de la última ejecución en MLflow
    try:
        latest_version = client.get_latest_versions(REGISTERED_MODEL_NAME)[-1]
        run_id = latest_version.run_id
        run_data = client.get_run(run_id).data.metrics
        
        promote_registered_model(REGISTERED_MODEL_NAME, run_data)
    except Exception as e:
        print(f" Error al ejecutar la promoción: {e}")
import mlflow
import mlflow.sklearn

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT_NAME = "bank-marketing-baseline"
#Se importa la libreria MLflow, se guarda modelos de sklearn, configura el logger. 

def configure_experiment(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> None:
    """Define bajo qué experimento de MLflow se agrupan todas las corridas."""
    mlflow.set_experiment(experiment_name)
#Le dice a MLflow -todo lo que registre a partir de ahora, agrúpalo bajo este nombre de experimento-.

def log_run(run_name: str, model, params: dict, metrics: dict) -> None:
    """Registra una corrida completa en MLflow: parámetros, métricas y modelo."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info("Run '%s' registrada en MLflow. Métricas: %s", run_name, metrics)

def log_all_results(results: list[dict]) -> None:
    """Recorre una lista de resultados (formato de train.py) y los loguea todos."""
    for result in results:
        log_run(
            run_name=result["name"],
            model=result["model"],
            params=result["params"],
            metrics=result["metrics"],
        )

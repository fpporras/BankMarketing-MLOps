"""
Este módulo no sabe entrenar nada — solo recibe un modelo ya entrenado
junto con sus parámetros y métricas, y lo registra en MLflow.
"""

import mlflow
import mlflow.sklearn

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT_NAME = "bank-marketing-baseline"


def configure_experiment(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> None:
    """Define bajo qué experimento de MLflow se agrupan todas las corridas."""
    mlflow.set_experiment(experiment_name)


def log_run(run_name: str, model, params: dict, metrics: dict) -> None:
    """Registra una corrida completa en MLflow: parámetros, métricas y modelo.

    No entrena nada — recibe el modelo ya entrenado (`model.fit()` ya se
    hizo antes, en train.py) y solo se encarga de dejarlo documentado.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info("Run '%s' registrada en MLflow. Métricas: %s", run_name, metrics)


def log_all_results(results: list[dict]) -> None:
    """Recorre una lista de resultados (formato de train.py) y los loguea todos.

    Cada elemento de `results` es un dict con las llaves:
    'name', 'model', 'params', 'metrics' — el mismo formato que devuelven
    train_baseline() y train_all_models() en train.py.
    """
    for result in results:
        log_run(
            run_name=result["name"],
            model=result["model"],
            params=result["params"],
            metrics=result["metrics"],
        )
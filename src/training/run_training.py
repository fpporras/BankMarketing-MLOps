"""


Orquestador: conecta train.py (entrenamiento) con tracking.py (MLflow).
Es el único archivo que importa ambos módulos — así train.py y tracking.py
se mantienen independientes entre sí.


"""

import argparse
import logging

import tracking
import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(data_path: str, test_size: float = 0.2, cv_folds: int = 5) -> None:
    # 1. Configurar MLflow (tracking.py)
    tracking.configure_experiment()

    # 2. Cargar y dividir los datos (train.py)
    df = train.load_data(data_path)
    X_train, X_test, y_train, y_test = train.make_split(df, test_size=test_size)

    # 3. Entrenar baseline + modelos (train.py)
    baseline_result = train.train_baseline(X_train, y_train, X_test, y_test)
    model_results = train.train_all_models(X_train, y_train, X_test, y_test, cv_folds=cv_folds)

    all_results = [baseline_result] + model_results

    # 4. Registrar todo en MLflow (tracking.py)
    tracking.log_all_results(all_results)

    # 5. Resumen en consola
    logger.info("=== Resumen de resultados ===")
    for result in all_results:
        m = result["metrics"]
        logger.info(
            "%-22s | acc=%.3f | prec=%.3f | rec=%.3f | f1=%.3f | roc_auc=%.3f",
            result["name"], m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento + tracking - Bank Marketing")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/bank_marketing_features.csv",
        help="Ruta al CSV con features ya procesadas (salida de Feature Engineering).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción del test set.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Número de folds para validación cruzada.")
    args = parser.parse_args()

    main(data_path=args.data, test_size=args.test_size, cv_folds=args.cv_folds)
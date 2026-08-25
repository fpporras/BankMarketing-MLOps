import argparse
import logging

import tracking
import train


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
# Se importan los archivos tracking.py y train.py que cree usando sus funciones y se configura el looger. 

def main(data_path: str, test_size: float = 0.2, cv_folds: int = 5) -> None:
    tracking.configure_experiment()

    df = train.load_data(data_path)
    X_train, X_test, y_train, y_test = train.make_split(df, test_size=test_size)

    baseline_result = train.train_baseline(X_train, y_train, X_test, y_test)
    model_results = train.train_all_models(X_train, y_train, X_test, y_test, cv_folds=cv_folds)

    all_results = [baseline_result] + model_results

    tracking.log_all_results(all_results)
#Se carga MLflow, divide los datos, entrena el baseline y los 3 modelos, junta todo en una sola lista, y se la entrega a tracking para que registre las 4 corridas en MLflow.

    logger.info("=== Resumen de resultados ===")
    for result in all_results:
        m = result["metrics"]
        logger.info(
            "%-22s | acc=%.3f | prec=%.3f | rec=%.3f | f1=%.3f | roc_auc=%.3f",
            result["name"], m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"],
        )
#Se recorre los resultado de baseline + los 3 modelos y por cada uno se imprime una linea ordenada con sus metricas para ver cual resultó mejor
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento + tracking - Bank Marketing")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/bank_marketing_features.csv",
        help="Ruta al CSV con features ya procesadas.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción del test set.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Número de folds para validación cruzada.")
    args = parser.parse_args()

    main(data_path=args.data, test_size=args.test_size, cv_folds=args.cv_folds)
#Se arma los argumentos que puedes pasar por terminal y en la ultima linea al llamar a Main hace que todo el script se ejecute.

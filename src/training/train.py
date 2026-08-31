# ============================================================
# TRAINING PIPELINE - BANK MARKETING MLOPS
# ============================================================

from pathlib import Path
import json
import tempfile
import warnings
import joblib

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# ============================================================
# IMPORTS DEL PROYECTO (MODULARES)
# ============================================================

from src.ingestion.ingest import load_raw_data
from src.validation.quality_gates import run_quality_gates
from src.features.build_features import build_features
from src.preprocessing.preprocess import build_preprocessor

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

RANDOM_STATE = 42
DATA_VERSION = "bank-marketing-v1"
EXPERIMENT_NAME = "bank-marketing-classification"
TARGET_COLUMN = "y"
TEST_SIZE = 0.20
CV_FOLDS = 5

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "bank_marketing.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MLARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
MLARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# EVALUACIÓN DEL MODELO
# ============================================================

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = np.nan

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else 0.0,
        "confusion_matrix": cm.tolist(),
    }

def print_metrics(title: str, metrics: dict) -> None:
    print("\n" + "=" * 45 + f"\n{title}\n" + "=" * 45)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"\nMatriz de confusión:\nVN={cm[0][0]} | FP={cm[0][1]}\nFN={cm[1][0]} | VP={cm[1][1]}")

# ============================================================
# REGISTRAR EXPERIMENTO EN MLFLOW
# ============================================================

def log_run(
    run_name: str,
    model,
    algorithm: str,
    params: dict,
    metrics: dict,
    feature_set: list,
    register_as: str = None,
) -> None:
    with mlflow.start_run(run_name=run_name):
        tracking_params = {
            "algorithm": algorithm,
            "random_seed": RANDOM_STATE,
            "data_version": DATA_VERSION,
            "feature_count": len(feature_set),
            "feature_set": ",".join(feature_set),
        }

        all_params = {**tracking_params, **params}
        safe_params = {str(k): str(v) for k, v in all_params.items()}
        mlflow.log_params(safe_params)

        numeric_metrics = {
            k: float(v) for k, v in metrics.items() 
            if k != "confusion_matrix" and isinstance(v, (int, float, np.integer, np.floating))
        }
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)

        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            mlflow.log_metric("true_negatives", int(cm[0][0]))
            mlflow.log_metric("false_positives", int(cm[0][1]))
            mlflow.log_metric("false_negatives", int(cm[1][0]))
            mlflow.log_metric("true_positives", int(cm[1][1]))

            display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no", "yes"])
            display.plot()
            plt.title(f"Matriz de Confusión - {run_name}")
            plt.tight_layout()
            
            # Guardar en local ANTES de cerrar la figura
            figures_dir = PROJECT_ROOT / "reports" / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(figures_dir / f"confusion_matrix_{run_name}.png")

            with tempfile.TemporaryDirectory() as temp_dir:
                cm_path = Path(temp_dir) / "confusion_matrix.png"
                plt.savefig(cm_path)
                plt.close()
                mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
            plt.close()       
            
        config = {
            "run_name": run_name,
            "algorithm": algorithm,
            "random_seed": RANDOM_STATE,
            "data_version": DATA_VERSION,
            "feature_set": feature_set,
            "feature_count": len(feature_set),
            "model_params": params,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False, default=str)
            mlflow.log_artifact(str(config_path), artifact_path="configuration")

        if register_as:
            mlflow.sklearn.log_model(sk_model=model, name="model", registered_model_name=register_as)
        else:
            mlflow.sklearn.log_model(sk_model=model, name="model")

# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\n" + "=" * 70 + "\nINICIANDO PIPELINE DE ENTRENAMIENTO\n" + "=" * 70)

    # 1. Carga de datos
    df_raw = load_raw_data()
    print(f"Datos cargados. Dimensiones: {df_raw.shape}")

    # 2. Quality Gates
    if not run_quality_gates(df_raw):
        raise SystemExit("Quality Gates fallidas. Pipeline bloqueado.")
    print("Quality Gates superados.")

    # 3. Feature Engineering (Usa la función modular importada)
    df_features = build_features(df_raw)
    print(f"Feature Engineering completado. Dimensiones: {df_features.shape}")

    # 4. Separación X / y
    if TARGET_COLUMN not in df_features.columns:
        raise ValueError(f"La columna target '{TARGET_COLUMN}' no existe.")

    X = df_features.drop(columns=[TARGET_COLUMN])
    FEATURE_SET = list(X.columns)

    # Convertir el target 'no'/'yes' a 0/1
    y = df_features[TARGET_COLUMN].map({"no": 0, "yes": 1}).astype(int)
    
    # 5. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 6. Baseline
    baseline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", DummyClassifier(strategy="most_frequent"))
    ])
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline, X_test, y_test)

    # 7. Modelos Candidatos
    pipelines = {
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, class_weight="balanced"))
        ]),
        "decision_tree": Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"))
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1))
        ]),
    }

    param_grids = {
        "logistic_regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
        "decision_tree": {"model__max_depth": [3, 5, 10, None], "model__min_samples_split": [2, 5, 10]},
        "random_forest": {"model__n_estimators": [200, 300, 400], "model__max_depth": [10, 15, 20, None]},
    }

    grid_searches = {}
    for name, pipeline in pipelines.items():
        print(f"\nAjustando hiperparámetros para {name}...")
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            scoring="f1",
            cv=CV_FOLDS,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)
        grid_searches[name] = grid_search

    # 8. Evaluación y Selección
    cv_results = {name: gs.best_score_ for name, gs in grid_searches.items()}
    best_model_name = max(cv_results, key=cv_results.get)
    best_grid_search = grid_searches[best_model_name]
    best_model = best_grid_search.best_estimator_

    print_metrics("BASELINE", baseline_metrics)
    final_metrics = evaluate_model(best_model, X_test, y_test)
    print_metrics(f"MODELO FINAL: {best_model_name}", final_metrics)

    # 9. Registro en MLflow
    log_run(
        run_name="baseline_dummy",
        model=baseline,
        algorithm="DummyClassifier",
        params={"strategy": "most_frequent"},
        metrics=baseline_metrics,
        feature_set=FEATURE_SET,
    )

    algorithm_mapping = {
        "logistic_regression": "LogisticRegression",
        "decision_tree": "DecisionTreeClassifier",
        "random_forest": "RandomForestClassifier",
    }

    for name, grid_search in grid_searches.items():
        log_run(
            run_name=f"{name}_tuned",
            model=grid_search.best_estimator_,
            algorithm=algorithm_mapping[name],
            params=grid_search.best_params_,
            metrics={"cv_f1": grid_search.best_score_},
            feature_set=FEATURE_SET,
        )

    log_run(
        run_name=f"{best_model_name}_final",
        model=best_model,
        algorithm=algorithm_mapping[best_model_name],
        params=best_grid_search.best_params_,
        metrics=final_metrics,
        feature_set=FEATURE_SET,
        register_as="bank-marketing-classifier",
    )

    print("\n" + "=" * 70 + "\nPIPELINE COMPLETADO CORRECTAMENTE\n" + "=" * 70)
    
    # Exportar el modelo final físicamente a models/
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, models_dir / "best_model.joblib")

if __name__ == "__main__":
    main()
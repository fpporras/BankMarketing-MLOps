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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.ingestion.ingest import load_raw_data
from src.validation.quality_gates import run_quality_gates
from src.features.prepare_data import prepare_processed_data
from src.preprocessing.preprocess import build_preprocessor
from src.evaluation.promote_model import promote_registered_model


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
DATA_VERSION = "bank-marketing-v1"

EXPERIMENT_NAME = "bank-marketing-classification"
REGISTERED_MODEL_NAME = "bank-marketing-classifier"

TARGET_COLUMN = "y"

TEST_SIZE = 0.20
CV_FOLDS = 5

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

    else:

        roc_auc = 0.0

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    return {
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),

        "precision": float(
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            )
        ),

        "roc_auc": float(roc_auc),

        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
    }


def print_metrics(name, metrics):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1       : {metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {metrics['roc_auc']:.4f}"
    )

    print(
        f"TN: {metrics['true_negatives']}"
    )

    print(
        f"FP: {metrics['false_positives']}"
    )

    print(
        f"FN: {metrics['false_negatives']}"
    )

    print(
        f"TP: {metrics['true_positives']}"
    )


# ============================================================
# MLFLOW
# ============================================================

def log_model_run(
    run_name,
    model,
    algorithm,
    params,
    metrics,
    feature_set,
    cv_f1=None,
    register_model=False,
):

    with mlflow.start_run(run_name=run_name):

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "algorithm",
            algorithm
        )

        mlflow.log_param(
            "random_seed",
            RANDOM_STATE
        )

        mlflow.log_param(
            "data_version",
            DATA_VERSION
        )

        mlflow.log_param(
            "feature_count",
            len(feature_set)
        )

        mlflow.log_param(
            "feature_set",
            ",".join(feature_set)
        )

        for key, value in params.items():

            mlflow.log_param(
                key,
                str(value)
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        if cv_f1 is not None:

            mlflow.log_metric(
                "cv_f1",
                float(cv_f1)
            )

        for key, value in metrics.items():

            mlflow.log_metric(
                key,
                float(value)
            )

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        cm = np.array([
            [
                metrics["true_negatives"],
                metrics["false_positives"]
            ],
            [
                metrics["false_negatives"],
                metrics["true_positives"]
            ]
        ])

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["no", "yes"]
        )

        display.plot()

        plt.title(
            f"Matriz de Confusión - {run_name}"
        )

        plt.tight_layout()

        figure_path = (
            REPORTS_DIR /
            f"confusion_matrix_{run_name}.png"
        )

        plt.savefig(
            figure_path
        )

        plt.close()

        mlflow.log_artifact(
            str(figure_path),
            artifact_path="evaluation"
        )

        # ----------------------------------------------------
        # Configuration artifact
        # ----------------------------------------------------

        config = {
            "run_name": run_name,
            "algorithm": algorithm,
            "random_state": RANDOM_STATE,
            "data_version": DATA_VERSION,
            "feature_set": feature_set,
            "feature_count": len(feature_set),
            "parameters": params,
            "metrics": metrics,
        }

        with tempfile.TemporaryDirectory() as temp_dir:

            config_path = (
                Path(temp_dir) /
                "config.json"
            )

            with open(
                config_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    config,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            mlflow.log_artifact(
                str(config_path),
                artifact_path="configuration"
            )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        if register_model:

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=REGISTERED_MODEL_NAME
            )

        else:

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        "\n" +
        "=" * 70 +
        "\nTRAINING PIPELINE\n" +
        "=" * 70
    )

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df_raw = load_raw_data()

    print(
        f"Raw data: {df_raw.shape}"
    )

    # --------------------------------------------------------
    # 2. Quality Gates
    # --------------------------------------------------------

    quality_passed = run_quality_gates(
        df_raw
    )

    if not quality_passed:

        raise SystemExit(
            "Quality Gates failed."
        )

    # --------------------------------------------------------
    # 3. Features
    # --------------------------------------------------------

    df_features = prepare_processed_data()
    
    # df_features = build_features(
    #    df_raw
    # )

    # --------------------------------------------------------
    # 4. X / y
    # --------------------------------------------------------

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

    feature_set = list(
        X.columns
    )

    # --------------------------------------------------------
    # 5. Train / Test
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # --------------------------------------------------------
    # 6. Baseline
    # --------------------------------------------------------

    baseline = Pipeline([
        (
            "preprocessor",
            build_preprocessor(X_train)
        ),

        (
            "model",
            DummyClassifier(
                strategy="most_frequent"
            )
        )
    ])

    baseline.fit(
        X_train,
        y_train
    )

    baseline_metrics = evaluate_model(
        baseline,
        X_test,
        y_test
    )

    print_metrics(
        "BASELINE",
        baseline_metrics
    )

    # --------------------------------------------------------
    # 7. Candidate Models
    # --------------------------------------------------------

    pipelines = {

        "logistic_regression": Pipeline([
            (
                "preprocessor",
                build_preprocessor(X_train)
            ),

            (
                "model",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=2000,
                    class_weight="balanced"
                )
            )
        ]),

        "decision_tree": Pipeline([
            (
                "preprocessor",
                build_preprocessor(X_train)
            ),

            (
                "model",
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced"
                )
            )
        ]),

        "random_forest": Pipeline([
            (
                "preprocessor",
                build_preprocessor(X_train)
            ),

            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1
                )
            )
        ])
    }

    param_grids = {

        "logistic_regression": {
            "model__C": [
                0.01,
                0.1,
                1,
                10,
                100
            ]
        },

        "decision_tree": {
            "model__max_depth": [
                3,
                5,
                10,
                None
            ],

            "model__min_samples_split": [
                2,
                5,
                10
            ]
        },

        "random_forest": {
            "model__n_estimators": [
                200,
                300
            ],

            "model__max_depth": [
                10,
                15,
                20,
                None
            ]
        }
    }

    # --------------------------------------------------------
    # 8. Training
    # --------------------------------------------------------

    results = {}

    for name, pipeline in pipelines.items():

        print(
            f"\nTraining {name}..."
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            scoring="f1",
            cv=CV_FOLDS,
            n_jobs=-1,
            refit=True
        )

        grid.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            grid.best_estimator_,
            X_test,
            y_test
        )

        results[name] = {
            "grid": grid,
            "metrics": metrics
        }

        print_metrics(
            name,
            metrics
        )

    # --------------------------------------------------------
    # 9. Select model
    # --------------------------------------------------------

    best_model_name = max(
        results,
        key=lambda name:
        results[name]["metrics"]["f1"]
    )

    best_grid = results[
        best_model_name
    ]["grid"]

    best_model = (
        best_grid.best_estimator_
    )

    best_metrics = (
        results[best_model_name]
        ["metrics"]
    )

    print_metrics(
        f"FINAL MODEL: {best_model_name}",
        best_metrics
    )

    # --------------------------------------------------------
    # 10. MLflow
    # --------------------------------------------------------

    log_model_run(
        run_name="baseline_dummy",
        model=baseline,
        algorithm="DummyClassifier",
        params={
            "strategy": "most_frequent"
        },
        metrics=baseline_metrics,
        feature_set=feature_set
    )

    algorithm_mapping = {

        "logistic_regression":
            "LogisticRegression",

        "decision_tree":
            "DecisionTreeClassifier",

        "random_forest":
            "RandomForestClassifier"
    }

    for name, result in results.items():

        log_model_run(
            run_name=f"{name}_candidate",
            model=result["grid"].best_estimator_,
            algorithm=algorithm_mapping[name],
            params=result["grid"].best_params_,
            metrics=result["metrics"],
            cv_f1=result["grid"].best_score_,
            feature_set=feature_set
        )

    # --------------------------------------------------------
    # 11. Register selected model
    # --------------------------------------------------------

    log_model_run(
        run_name=f"{best_model_name}_production_candidate",
        model=best_model,
        algorithm=algorithm_mapping[
            best_model_name
        ],
        params=best_grid.best_params_,
        metrics=best_metrics,
        cv_f1=best_grid.best_score_,
        feature_set=feature_set,
        register_model=True
    )
    
    promote_registered_model(
    model_name="bank-marketing-classifier", 
    metrics=best_metrics
    )

    # --------------------------------------------------------
    # 12. Save model locally
    # --------------------------------------------------------

    model_path = (
        MODELS_DIR /
        "best_model.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print(
        f"\nModelo guardado en: {model_path}"
    )

    print(
        "\nTRAINING COMPLETED"
    )


if __name__ == "__main__":
    main()
from pathlib import Path
import json
import tempfile
import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.compose import ColumnTransformer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DATA_VERSION = (
    "bank-marketing-v1"
)

EXPERIMENT_NAME = (
    "bank-marketing-classification"
)

REGISTERED_MODEL_NAME = (
    "bank-marketing-classifier"
)

TARGET_COLUMN = "y"

TEST_SIZE = 0.20

CV_FOLDS = 5

MLFLOW_TRACKING_URI = (
    "http://127.0.0.1:5000"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

MODELS_DIR = (
    PROJECT_ROOT /
    "models"
)

REPORTS_DIR = (
    PROJECT_ROOT /
    "reports" /
    "figures"
)


MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(X):
    """
    Construye el preprocesador para variables numéricas
    y categóricas.
    """

    numerical_columns = (
        X.select_dtypes(
            include=["int64", "float64"]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            include=["object"]
        )
        .columns
        .tolist()
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numerical_columns,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns,
            ),
        ]
    )

    return preprocessor


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evalúa un modelo de clasificación.
    """

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Probabilidades
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        y_prob = (
            model
            .predict_proba(X_test)[:, 1]
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

    else:

        roc_auc = 0.0

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    return {

        "accuracy": float(
            accuracy_score(
                y_test,
                y_pred
            )
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

        "roc_auc": float(
            roc_auc
        ),

        "true_negatives": int(
            cm[0][0]
        ),

        "false_positives": int(
            cm[0][1]
        ),

        "false_negatives": int(
            cm[1][0]
        ),

        "true_positives": int(
            cm[1][1]
        ),
    }


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    name,
    metrics
):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{metrics['roc_auc']:.4f}"
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
# MLFLOW LOGGING
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
    """
    Registra un modelo, parámetros, métricas,
    configuración y matriz de confusión en MLflow.
    """

    with mlflow.start_run(
        run_name=run_name
    ):

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
        # CV metric
        # ----------------------------------------------------

        if cv_f1 is not None:

            mlflow.log_metric(
                "cv_f1",
                float(cv_f1)
            )

        # ----------------------------------------------------
        # Evaluation metrics
        # ----------------------------------------------------

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
                metrics[
                    "true_negatives"
                ],

                metrics[
                    "false_positives"
                ]
            ],

            [
                metrics[
                    "false_negatives"
                ],

                metrics[
                    "true_positives"
                ]
            ]
        ])

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "no",
                "yes"
            ]
        )

        display.plot()

        plt.title(
            f"Matriz de Confusión - "
            f"{run_name}"
        )

        plt.tight_layout()

        figure_path = (
            REPORTS_DIR /
            f"confusion_matrix_"
            f"{run_name}.png"
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

            "run_name":
                run_name,

            "algorithm":
                algorithm,

            "random_state":
                RANDOM_STATE,

            "data_version":
                DATA_VERSION,

            "feature_set":
                feature_set,

            "feature_count":
                len(feature_set),

            "parameters":
                params,

            "metrics":
                metrics,
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

                registered_model_name=(
                    REGISTERED_MODEL_NAME
                )
            )

        else:

            mlflow.sklearn.log_model(

                sk_model=model,

                name="model"
            )


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_models(
    df_features
):
    """
    Entrena y evalúa los modelos candidatos.

    Esta función NO realiza:
        - Ingesta
        - Data Quality
        - Quality Gates
        - Feature Engineering

    Esas etapas son responsabilidad del orchestrator.

    Returns
    -------
    dict
        Resultados completos del entrenamiento.
    """

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    # ========================================================
    # 1. X / y
    # ========================================================

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

    print(
        f"\nFeatures: {X.shape}"
    )

    print(
        f"Target: {y.shape}"
    )

    print(
        "\nDistribución del target:"
    )

    print(
        y.value_counts()
    )

    # ========================================================
    # 2. TRAIN / TEST
    # ========================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,
            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y
        )
    )

    print(
        "\nTrain:"
        f" {X_train.shape}"
    )

    print(
        "Test:"
        f" {X_test.shape}"
    )

    # ========================================================
    # 3. BASELINE
    # ========================================================

    print(
        "\nEntrenando baseline..."
    )

    baseline = Pipeline([

        (
            "preprocessor",
            build_preprocessor(
                X_train
            )
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

    baseline_metrics = (
        evaluate_model(
            baseline,
            X_test,
            y_test
        )
    )

    print_metrics(
        "BASELINE",
        baseline_metrics
    )

    # ========================================================
    # 4. CANDIDATE MODELS
    # ========================================================

    pipelines = {

        "logistic_regression":

            Pipeline([

                (
                    "preprocessor",
                    build_preprocessor(
                        X_train
                    )
                ),

                (
                    "model",
                    LogisticRegression(

                        random_state=
                            RANDOM_STATE,

                        max_iter=2000,

                        class_weight=
                            "balanced"
                    )
                )
            ]),

        "decision_tree":

            Pipeline([

                (
                    "preprocessor",
                    build_preprocessor(
                        X_train
                    )
                ),

                (
                    "model",
                    DecisionTreeClassifier(

                        random_state=
                            RANDOM_STATE,

                        class_weight=
                            "balanced"
                    )
                )
            ]),

        "random_forest":

            Pipeline([

                (
                    "preprocessor",
                    build_preprocessor(
                        X_train
                    )
                ),

                (
                    "model",
                    RandomForestClassifier(

                        random_state=
                            RANDOM_STATE,

                        class_weight=
                            "balanced",

                        n_jobs=-1
                    )
                )
            ])
    }

    # ========================================================
    # 5. PARAMETER GRIDS
    # ========================================================

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

    # ========================================================
    # 6. TRAINING / GRID SEARCH
    # ========================================================

    results = {}

    for name, pipeline in pipelines.items():

        print(
            "\n" +
            "-" * 70
        )

        print(
            f"Training: {name}"
        )

        print(
            "-" * 70
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

        metrics = (
            evaluate_model(

                grid.best_estimator_,

                X_test,

                y_test
            )
        )

        results[name] = {

            "grid":
                grid,

            "metrics":
                metrics
        }

        print(
            "\nBest parameters:"
        )

        print(
            grid.best_params_
        )

        print_metrics(
            name,
            metrics
        )

    # ========================================================
    # 7. SELECT BEST MODEL
    # ========================================================

    best_model_name = max(

        results,

        key=lambda name:
            results[name]
            ["metrics"]
            ["f1"]
    )

    best_grid = (
        results[
            best_model_name
        ]["grid"]
    )

    best_model = (
        best_grid.best_estimator_
    )

    best_metrics = (
        results[
            best_model_name
        ]["metrics"]
    )

    print(
        "\n" +
        "=" * 70
    )

    print(
        "FINAL MODEL"
    )

    print(
        "=" * 70
    )

    print(
        f"Modelo seleccionado: "
        f"{best_model_name}"
    )

    print_metrics(
        "BEST MODEL",
        best_metrics
    )

    # ========================================================
    # 8. ALGORITHM MAPPING
    # ========================================================

    algorithm_mapping = {

        "logistic_regression":
            "LogisticRegression",

        "decision_tree":
            "DecisionTreeClassifier",

        "random_forest":
            "RandomForestClassifier"
    }

    # ========================================================
    # 9. MLFLOW
    # ========================================================

    print(
        "\n" +
        "=" * 70
    )

    print(
        "MLFLOW LOGGING"
    )

    print(
        "=" * 70
    )

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    log_model_run(

        run_name=
            "baseline_dummy",

        model=
            baseline,

        algorithm=
            "DummyClassifier",

        params={
            "strategy":
                "most_frequent"
        },

        metrics=
            baseline_metrics,

        feature_set=
            feature_set
    )

    # --------------------------------------------------------
    # Candidate models
    # --------------------------------------------------------

    for name, result in results.items():

        log_model_run(

            run_name=
                f"{name}_candidate",

            model=
                result["grid"]
                .best_estimator_,

            algorithm=
                algorithm_mapping[name],

            params=
                result["grid"]
                .best_params_,

            metrics=
                result["metrics"],

            cv_f1=
                result["grid"]
                .best_score_,

            feature_set=
                feature_set
        )

    # ========================================================
    # 10. REGISTER BEST MODEL
    # ========================================================

    print(
        "\n" +
        "=" * 70
    )

    print(
        "REGISTERING BEST MODEL"
    )

    print(
        "=" * 70
    )

    log_model_run(

        run_name=
            f"{best_model_name}_"
            f"production_candidate",

        model=
            best_model,

        algorithm=
            algorithm_mapping[
                best_model_name
            ],

        params=
            best_grid.best_params_,

        metrics=
            best_metrics,

        cv_f1=
            best_grid.best_score_,

        feature_set=
            feature_set,

        register_model=True
    )

    print(
        "\n✓ Modelo registrado en MLflow."
    )

    def save_model(model=None, path=None):
        # Si el orquestador no pasa ruta o modelo, usa los por defecto
        target_model = model if model is not None else best_model
        target_path = path if path is not None else (MODELS_DIR / "best_model.joblib")

        joblib.dump(target_model, target_path)
        print(f"✓ Modelo guardado localmente en: {target_path}")
        
    # ========================================================
    # 11. RETURN RESULTS
    # ========================================================

    return {

        "best_model_name":
            best_model_name,

        "best_model":
            best_model,

        "best_metrics":
            best_metrics,

        "best_grid":
            best_grid,

        "feature_set":
            feature_set,

        "algorithm_mapping":
            algorithm_mapping,

        "results":
            results,

        "baseline_metrics":
            baseline_metrics,
            
        "registered":
            True,
            
        "save_model": save_model,  # Pasa la referencia de la función
    }

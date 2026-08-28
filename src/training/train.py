
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

from src.ingestion.ingest import load_raw_data
from src.validation.quality_gates import run_quality_gates
from src.features.build_features import build_features
from src.preprocessing.preprocess import build_preprocessor


# ============================================================
# CONFIGURACIÓN
# ============================================================

RANDOM_STATE = 42


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():

    # ========================================================
    # 1. INGESTA
    # ========================================================

    df_raw = load_raw_data()

    print("Datos cargados correctamente.")
    print(f"Dimensiones originales: {df_raw.shape}")


    # ========================================================
    # 2. QUALITY GATES
    # ========================================================

    quality_passed = run_quality_gates(df_raw)

    if not quality_passed:
        raise SystemExit(
            "El pipeline fue bloqueado por calidad de datos."
        )

    print("\nQuality Gates superados correctamente.")


    # ========================================================
    # 3. FEATURE ENGINEERING
    # ========================================================

    df_features = build_features(df_raw)

    print("\nFeature Engineering completado.")
    print(f"Dimensiones finales: {df_features.shape}")

    print("\nColumnas:")
    print(df_features.columns.tolist())

    print("\nPrimeras filas:")
    print(df_features.head())


    # ========================================================
    # 4. SEPARAR X E y
    # ========================================================

    X = df_features.drop(
        columns=["y"]
    )

    y = df_features["y"].copy()


    # Convertir yes/no a 1/0
    if y.dtype == object:

        y = (
            y.astype(str)
            .str.strip()
            .str.lower()
            .map({
                "yes": 1,
                "no": 0
            })
        )


    # Validar target
    if y.isna().any():

        raise ValueError(
            "La variable objetivo contiene valores no reconocidos."
        )


    y = y.astype(int)


    print("\nSeparación X / y completada.")

    print(
        f"Dimensiones de X: {X.shape}"
    )

    print(
        f"Dimensiones de y: {y.shape}"
    )


    print("\nDistribución de clases:")

    print(
        y.value_counts()
    )


    print("\nDistribución porcentual:")

    print(
        y.value_counts(
            normalize=True
        )
    )


    # ========================================================
    # 5. TRAIN / TEST
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=RANDOM_STATE,

        stratify=y

    )


    print("\nTrain/Test Split completado.")

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test: {X_test.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"y_test: {y_test.shape}"
    )


    # ========================================================
    # 6. PREPROCESADOR
    # ========================================================
  
    # Solo construimos el preprocesador.
    # NO hacemos fit_transform aquí.
    # El Pipeline hará el entrenamiento
    # del preprocesador dentro de cada fold.
    # ========================================================

    preprocessor = build_preprocessor(
        X_train
    )


    print(
        "\nPreprocesador construido correctamente."
    )


    # ========================================================
    # 7. BASELINE
    # ========================================================

    baseline = DummyClassifier(

        strategy="most_frequent",

        random_state=RANDOM_STATE

    )


    baseline.fit(
        X_train,
        y_train
    )


    print(
        "\nBaseline entrenado correctamente."
    )


    # ========================================================
    # 8. MODELOS
    # ========================================================

    models = {

        "logistic_regression": LogisticRegression(

            max_iter=1000,

            random_state=RANDOM_STATE

        ),


        "decision_tree": DecisionTreeClassifier(

            random_state=RANDOM_STATE

        ),


        "random_forest": RandomForestClassifier(

            n_estimators=100,

            random_state=RANDOM_STATE,

            n_jobs=1

        )

    }


    # ========================================================
    # 9. PIPELINES
    # ========================================================
  
    # SMOTE solamente se aplica al TRAIN
    # de cada fold.
    # ========================================================

    pipelines = {}


    for name, model in models.items():

        pipelines[name] = Pipeline(

            steps=[

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE
                    )
                ),

                (
                    "model",
                    model
                )

            ]

        )


    print(
        "\nPipelines construidos correctamente."
    )

    print(
        "Preprocesamiento -> SMOTE -> Modelo"
    )


    # ========================================================
    # 10. HIPERPARÁMETROS
    # ========================================================

    param_grids = {

        # ----------------------------------------------------
        # LOGISTIC REGRESSION
        # ----------------------------------------------------

        "logistic_regression": {

            "model__C": [
                0.01,
                0.1,
                1,
                10,
                100
            ]

        },


        # ----------------------------------------------------
        # DECISION TREE
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------
      

        "random_forest": {

            "model__n_estimators": [
                200,
                300,
                400
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
    # 11. CROSS VALIDATION
    # ========================================================

    skf = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=RANDOM_STATE

    )


    # ========================================================
    # 12. GRID SEARCH
    # ========================================================

    print(
        "\n============================================="
    )

    print(
        "INICIANDO AJUSTE DE HIPERPARÁMETROS"
    )

    print(
        "============================================="
    )


    grid_searches = {}

    cv_results = {}


    for name, pipeline in pipelines.items():

        print(
            f"\nBuscando mejores hiperparámetros "
            f"para {name}..."
        )


        grid_search = GridSearchCV(

            estimator=pipeline,

            param_grid=param_grids[name],

            # F1 es nuestra métrica de selección
            scoring="f1",

            cv=skf,

            # 1 evita el problema de BrokenProcessPool
            n_jobs=1,

            refit=True,

            # Muestra el progreso en terminal
            verbose=2

        )


        # ====================================================
        # ENTRENAMIENTO
        # ====================================================
        #
        # NO usamos X_train_bal.
        # GridSearch recibe los datos originales.
        #
        # Dentro de cada fold ocurre:
        # 1.Preprocesamiento
        # 2.SMOTE
        # 3.Modelo
        # ====================================================

        grid_search.fit(
            X_train,
            y_train
        )


        grid_searches[name] = (
            grid_search
        )


        cv_results[name] = {

            "best_cv_f1":
                grid_search.best_score_,

            "best_params":
                grid_search.best_params_

        }


        print(
            f"\nMejores parámetros para {name}: "
            f"{grid_search.best_params_}"
        )


        print(
            f"Mejor F1 en validación cruzada: "
            f"{grid_search.best_score_:.4f}"
        )


    # ========================================================
    # 13. RESULTADOS DE CROSS VALIDATION
    # ========================================================

    print(
        "\n============================================="
    )

    print(
        "RESULTADOS DE CROSS VALIDATION"
    )

    print(
        "============================================="
    )


    for name, result in cv_results.items():

        print(

            f"{name:22s} | "

            f"F1 CV = "
            f"{result['best_cv_f1']:.4f}"

        )


    # ========================================================
    # 14. SELECCIONAR MEJOR MODELO
    # ========================================================
    #
    # El ganador se selecciona por F1
    # de Cross Validation.
    #
    # NO utilizamos TEST para escogerlo.
    # ========================================================

    best_model_name = max(

        cv_results,

        key=lambda name:
            cv_results[name]["best_cv_f1"]

    )


    best_grid_search = (
        grid_searches[
            best_model_name
        ]
    )


    best_model = (
        best_grid_search.best_estimator_
    )


    best_params = (
        best_grid_search.best_params_
    )


    best_cv_f1 = (
        best_grid_search.best_score_
    )


    print(
        "\n============================================="
    )

    print(
        "MEJOR MODELO SEGÚN CROSS VALIDATION"
    )

    print(
        "============================================="
    )


    print(
        f"Modelo seleccionado: "
        f"{best_model_name}"
    )


    print(
        f"F1 promedio CV: "
        f"{best_cv_f1:.4f}"
    )


    print(
        f"Mejores parámetros: "
        f"{best_params}"
    )


    # ========================================================
    # 15. EVALUACIÓN FINAL EN TEST
    # ========================================================

    final_metrics = evaluate_model(

        best_model,

        X_test,

        y_test

    )


    # ========================================================
    # 16. EVALUACIÓN DEL BASELINE
    # ========================================================

    baseline_metrics = evaluate_model(

        baseline,

        X_test,

        y_test

    )


    # ========================================================
    # 17. RESULTADOS BASELINE
    # ========================================================

    print(
        "\n============================================="
    )

    print(
        "BASELINE"
    )

    print(
        "============================================="
    )


    print_metrics(
        baseline_metrics
    )


    # ========================================================
    # 18. RESULTADOS MODELO FINAL
    # ========================================================

    print(
        "\n============================================="
    )

    print(
        f"MODELO FINAL: {best_model_name}"
    )

    print(
        "============================================="
    )


    print_metrics(
        final_metrics
    )


    # ========================================================
    # 19. MLFLOW
    # ========================================================

    print(
        "\n============================================="
    )

    print(
        "REGISTRANDO EXPERIMENTOS EN MLFLOW"
    )

    print(
        "============================================="
    )


    mlflow.set_experiment(
        "bank-marketing-classification"
    )


    # ========================================================
    # REGISTRAR BASELINE
    # ========================================================

    log_run(

        run_name="baseline_dummy",

        model=baseline,

        params={
            "strategy": "most_frequent"
        },

        metrics=baseline_metrics

    )


    # ========================================================
    # REGISTRAR MODELOS DEL GRIDSEARCH
    # ========================================================

    for name, grid_search in grid_searches.items():

        log_run(

            run_name=f"{name}_tuned",

            model=grid_search.best_estimator_,

            params=grid_search.best_params_,

            metrics={
                "cv_f1":
                    grid_search.best_score_
            }

        )


    # ========================================================
    # REGISTRAR MODELO FINAL
    # ========================================================

    log_run(

        run_name=f"{best_model_name}_final",

        model=best_model,

        params=best_params,

        metrics=final_metrics,

        register_as="bank-marketing-classifier"

    )


    # ========================================================
    # FIN
    # ========================================================

    print(
        "\nPipeline completado correctamente."
    )


    return {

        "best_model_name":
            best_model_name,

        "best_cv_f1":
            best_cv_f1,

        "best_params":
            best_params,

        "test_metrics":
            final_metrics

    }


# ============================================================
# FUNCIÓN DE EVALUACIÓN
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
) -> dict:

    """
    Calcula las métricas de clasificación.
    """


    # ========================================================
    # PREDICCIONES
    # ========================================================

    y_pred = model.predict(
        X_test
    )


    # ========================================================
    # PROBABILIDADES
    # ========================================================

    y_proba = model.predict_proba(
        X_test
    )[:, 1]


    # ========================================================
    # MATRIZ DE CONFUSIÓN
    # ========================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    # ========================================================
    # MÉTRICAS
    # ========================================================

    return {

        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            y_proba
        ),

        "confusion_matrix":
            cm.tolist()

    }


# ============================================================
# FUNCIÓN PARA MOSTRAR MÉTRICAS
# ============================================================

def print_metrics(
    metrics
):

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
        f"F1 Score : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{metrics['roc_auc']:.4f}"
    )


    # ========================================================
    # MATRIZ DE CONFUSIÓN
    # ========================================================

    cm = metrics[
        "confusion_matrix"
    ]


    print(
        "\nMatriz de confusión:"
    )


    print(
        f"VN={cm[0][0]} | "
        f"FP={cm[0][1]}"
    )


    print(
        f"FN={cm[1][0]} | "
        f"VP={cm[1][1]}"
    )


# ============================================================
# FUNCIÓN MLFLOW
# ============================================================
# Se centraliza el registro de experimentos de Machine Learning.
# Ademas se guarda hiperparámetros, métricas y modelos entrenados en MLflow.


def log_run(
    run_name: str,
    model,
    params: dict,
    metrics: dict,
    register_as: str = None
) -> None:

    """
    Registra parámetros, métricas y modelo
    en MLflow.
    """


    with mlflow.start_run(
        run_name=run_name
    ):


        # ====================================================
        # PARÁMETROS
        # ====================================================

        mlflow.log_params(
            params
        )


        # ====================================================
        # MÉTRICAS
        # ====================================================

        numeric_metrics = {

            key: value

            for key, value in metrics.items()

            if key != "confusion_matrix"

        }


        mlflow.log_metrics(
            numeric_metrics
        )


        # ====================================================
        # MATRIZ DE CONFUSIÓN
        # ====================================================

        if "confusion_matrix" in metrics:

            cm = metrics[
                "confusion_matrix"
            ]


            mlflow.log_metric(
                "true_negatives",
                cm[0][0]
            )


            mlflow.log_metric(
                "false_positives",
                cm[0][1]
            )


            mlflow.log_metric(
                "false_negatives",
                cm[1][0]
            )


            mlflow.log_metric(
                "true_positives",
                cm[1][1]
            )

        # ====================================================
        # GUARDAR / REGISTRAR MODELO
        # ====================================================

        trusted_types = [
            "imblearn.pipeline.Pipeline",
            "imblearn.over_sampling._smote.base.SMOTE"
        ]

        if register_as:

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=register_as,
                skops_trusted_types=trusted_types
            )

        else:

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                skops_trusted_types=trusted_types
            )
# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    main()
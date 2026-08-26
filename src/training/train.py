
from sklearn.model_selection import train_test_split
from src.ingestion.ingest import load_raw_data
from src.validation.quality_gates import run_quality_gates
from src.features.build_features import build_features
from src.preprocessing.preprocess import build_preprocessor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
    confusion_matrix,
)

import mlflow
import mlflow.sklearn


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
    y = df_features["y"]

    if y.dtype == object:
        y = y.map({"yes": 1, "no": 0})

    print("\nSeparación X / y completada.")
    print(f"Dimensiones de X: {X.shape}")
    print(f"Dimensiones de y: {y.shape}")

    # ========================================================
    # 5. TRAIN / TEST SPLIT (80% entrenamiento / 20% prueba)
    # ========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print("\nTrain/Test Split completado.")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    # ========================================================
    # 6. CONSTRUIR PREPROCESADOR
    # ========================================================
    preprocessor = build_preprocessor(
        X_train
    )
    print("\nPreprocesador construido correctamente.")

    # ========================================================
    # 7. APLICAR PREPROCESAMIENTO
    # ========================================================
    X_train_processed = preprocessor.fit_transform(
        X_train
    )
    X_test_processed = preprocessor.transform(
        X_test
    )
    print("\nPreprocesamiento aplicado correctamente.")
    print(f"X_train procesado: {X_train_processed.shape}")
    print(f"X_test procesado: {X_test_processed.shape}")

    # ========================================================
    # 8. ENTRENAR BASELINE
    # ========================================================
    baseline = DummyClassifier(strategy="most_frequent", random_state=42)
    baseline.fit(X_train_processed, y_train)
    baseline_metrics = evaluate_model(baseline, X_test_processed, y_test)
    print("\nBaseline entrenado.")
    print(f"Métricas baseline: {baseline_metrics}")

    # ========================================================
    # 9. ENTRENAR LOS 3 MODELOS DE CLASIFICACIÓN
    # ========================================================
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
    }

    results = {"baseline_dummy": baseline_metrics}

    for name, model in models.items():
        print(f"\nEntrenando {name}...")
        model.fit(X_train_processed, y_train)
        metrics = evaluate_model(model, X_test_processed, y_test)
        results[name] = metrics
        print(f"Métricas {name}: {metrics}")

    # ========================================================
    # 10. RESUMEN FINAL
    # ========================================================
    print("\n=== Resumen de resultados ===")
    for name, metrics in results.items():
        print(
            f"{name:22s} | acc={metrics['accuracy']:.3f} | "
            f"prec={metrics['precision']:.3f} | rec={metrics['recall']:.3f} | "
            f"f1={metrics['f1']:.3f} | roc_auc={metrics['roc_auc']:.3f}"
        )
        cm = metrics["confusion_matrix"]
        print(f"    Matriz de confusión: [[VN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, VP={cm[1][1]}]]")

    # ========================================================
    # 11. AJUSTE DEL MODELO CON HIPERPARÁMETROS
    # ========================================================
    print("\n\n=== Iniciando ajuste de hiperparámetros (GridSearchCV) ===")

    base_models_for_tuning = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", random_state=42
        ),
    }

    param_grids = {
        "logistic_regression": {
            "C": [0.01, 0.1, 1, 10, 100],
        },
        "decision_tree": {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
        },
        "random_forest": {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [8, 10, 12, None],
},
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_models = {}
    best_params_per_model = {}

    for name, base_model in base_models_for_tuning.items():
        print(f"\nBuscando mejores hiperparámetros para {name}...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grids[name],
            scoring="f1",
            cv=skf,
            n_jobs=-1,
        )
        grid_search.fit(X_train_processed, y_train)

        best_models[name] = grid_search.best_estimator_
        best_params_per_model[name] = grid_search.best_params_
        print(f"Mejores parámetros para {name}: {grid_search.best_params_}")
        print(f"Mejor F1 en validación cruzada: {grid_search.best_score_:.4f}")

    # ========================================================
    # 12. EVALUAR MODELOS AJUSTADOS
    # ========================================================
    tuned_results = {}
    for name, model in best_models.items():
        metrics = evaluate_model(model, X_test_processed, y_test)
        tuned_results[name] = metrics
        print(f"\nMétricas {name} (con mejores hiperparámetros): {metrics}")

    # ========================================================
    # 13. RESUMEN COMPARATIVO (sin tuning vs. con tuning)
    # ========================================================
    print("\n=== Comparación: Sin tuning vs. Con tuning ===")
    for name in models.keys():
        print(f"{name}:")
        print(f"  Sin tuning  -> f1={results[name]['f1']:.3f} | roc_auc={results[name]['roc_auc']:.3f}")
        print(f"  Con tuning  -> f1={tuned_results[name]['f1']:.3f} | roc_auc={tuned_results[name]['roc_auc']:.3f}")

    # ========================================================
    # 14. REGISTRAR TODO EN MLFLOW
    # ========================================================
    print("\n=== Registrando corridas en MLflow ===")
    mlflow.set_experiment("bank-marketing-classification")

    all_runs = [
        ("baseline_dummy", baseline, {"strategy": "most_frequent"}, baseline_metrics),
    ]

    for name, model in models.items():
        all_runs.append((name, model, model.get_params(), results[name]))

    for name, model in best_models.items():
        all_runs.append((f"{name}_tuned", model, best_params_per_model[name], tuned_results[name]))

    for run_name, model, params, metrics in all_runs:
        if run_name == "random_forest_tuned":
            log_run(run_name, model, params, metrics, register_as="bank-marketing-classifier")
        else:
            log_run(run_name, model, params, metrics)

    return results


def evaluate_model(model, X_test, y_test) -> dict:
    """Calcula métricas relevantes para un problema de clasificación desbalanceado."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def log_run(run_name: str, model, params: dict, metrics: dict, register_as: str = None) -> None:
    """Registra una corrida completa en MLflow: parámetros, métricas y modelo.

    Args:
        run_name: Nombre de la corrida.
        model: Modelo entrenado.
        params: Parámetros del modelo.
        metrics: Métricas de evaluación.
        register_as: Nombre bajo el cual registrar el modelo en MLflow.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        numeric_metrics = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        mlflow.log_metrics(numeric_metrics)

        cm = metrics["confusion_matrix"]
        mlflow.log_metric("true_negatives", cm[0][0])
        mlflow.log_metric("false_positives", cm[0][1])
        mlflow.log_metric("false_negatives", cm[1][0])
        mlflow.log_metric("true_positives", cm[1][1])

        if register_as:
            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=register_as)
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Run '{run_name}' registrada en MLflow.")


if __name__ == "__main__":
    main()
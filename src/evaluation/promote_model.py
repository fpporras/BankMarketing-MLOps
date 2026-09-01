import mlflow

from mlflow.tracking import MlflowClient


# ============================================================
# CONFIGURATION
# ============================================================

MIN_F1 = 0.40

MIN_RECALL = 0.55

MIN_ROC_AUC = 0.70


MLFLOW_TRACKING_URI = (
    "http://127.0.0.1:5000"
)

REGISTERED_MODEL_NAME = (
    "bank-marketing-classifier"
)


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_model(
    metrics: dict
) -> bool:
    """
    Verifica si el modelo cumple con
    los umbrales mínimos establecidos.
    """

    checks = {

        "f1":
            metrics.get(
                "f1",
                0
            ) >= MIN_F1,

        "recall":
            metrics.get(
                "recall",
                0
            ) >= MIN_RECALL,

        "roc_auc":
            metrics.get(
                "roc_auc",
                0
            ) >= MIN_ROC_AUC,
    }

    print(
        "\n" +
        "=" * 50
    )

    print(
        "MODEL VALIDATION"
    )

    print(
        "=" * 50
    )

    for metric, passed in checks.items():

        value = metrics.get(
            metric,
            0.0
        )

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{metric.upper():<8}: "
            f"{value:.4f} -> {status}"
        )

    return all(
        checks.values()
    )


# ============================================================
# GET LATEST REGISTERED VERSION
# ============================================================

def get_latest_registered_version(
    model_name: str
):
    """
    Obtiene la versión más reciente
    del modelo registrado.
    """

    client = MlflowClient(
        tracking_uri=
            MLFLOW_TRACKING_URI
    )

    versions = (
        client
        .search_model_versions(
            filter_string=
                f"name='{model_name}'"
        )
    )

    if not versions:

        return None

    versions = sorted(

        versions,

        key=lambda version:
            int(version.version)
    )

    return versions[-1]


# ============================================================
# PROMOTION
# ============================================================

def promote_registered_model(
    model_name: str,
    metrics: dict
):
    """
    Valida las métricas y, si cumplen
    los umbrales, asigna el alias
    'champion' a la última versión registrada.
    """

    # ========================================================
    # 1. VALIDATE
    # ========================================================

    if not validate_model(
        metrics
    ):

        print(
            "\nEl modelo NO cumplió "
            "los umbrales requeridos."
        )

        print(
            "No será promocionado."
        )

        return False

    # ========================================================
    # 2. MLflow Client
    # ========================================================

    client = MlflowClient(
        tracking_uri=
            MLFLOW_TRACKING_URI
    )

    # ========================================================
    # 3. Latest Version
    # ========================================================

    latest_version = (
        get_latest_registered_version(
            model_name
        )
    )

    if latest_version is None:

        print(
            f"\nNo se encontraron "
            f"versiones registradas para "
            f"'{model_name}'."
        )

        return False

    version_number = (
        latest_version.version
    )

    print(
        "\nVersión registrada encontrada:"
    )

    print(
        f"Modelo: {model_name}"
    )

    print(
        f"Versión: {version_number}"
    )

    print(
        f"Run ID: "
        f"{latest_version.run_id}"
    )

    # ========================================================
    # 4. Assign Champion Alias
    # ========================================================

    client.set_registered_model_alias(

        name=model_name,

        alias="champion",

        version=version_number
    )

    # ========================================================
    # 5. Success
    # ========================================================

    print(
        "\n" +
        "=" * 50
    )

    print(
        "MODEL PROMOTION"
    )

    print(
        "=" * 50
    )

    print(
        f"✓ Modelo: {model_name}"
    )

    print(
        f"✓ Versión: {version_number}"
    )

    print(
        "✓ Alias: champion"
    )

    print(
        "=" * 50
    )

    return True

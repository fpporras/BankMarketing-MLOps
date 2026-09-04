from pathlib import Path

import pandas as pd

from src.monitoring.data_monitor import (
    monitor_numeric_features
)

from src.monitoring.model_monitor import (
    evaluate_production_batch
)

from src.monitoring.system_monitor import (
    monitor_system
)

from src.retraining.retrain_decision import (
    should_retrain
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "reference.csv"
)

PRODUCTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "production"
)


NUMERIC_COLUMNS = [

    "age",

    "balance",

    "campaign",

    "pdays",

    "previous"
]


def get_latest_batch():
    """
    Obtiene el batch de producción más reciente.
    """

    batches = sorted(
        PRODUCTION_DIR.glob(
            "batch_*.csv"
        )
    )

    if not batches:

        raise FileNotFoundError(
            "No se encontraron production batches en: "
            f"{PRODUCTION_DIR}"
        )

    return batches[-1]


def run_monitoring():
    """
    Ejecuta todo el proceso de monitoring.

    Returns
    -------
    dict
        Resultado completo del monitoring.
    """

    print("\n" + "=" * 80)
    print("MONITORING - BANK MARKETING")
    print("=" * 80)

    # ========================================================
    # VALIDATE FILES
    # ========================================================

    if not REFERENCE_FILE.exists():

        raise FileNotFoundError(
            "No existe el reference dataset.\n"
            "Ejecuta primero:\n"
            "python -m src.monitoring.create_reference"
        )

    batch_file = get_latest_batch()

    # ========================================================
    # O1 - SYSTEM MONITORING
    # ========================================================

    system_report = monitor_system()

    print("\nO1 - SYSTEM MONITORING")

    print(
        system_report.to_string(
            index=False
        )
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    reference_df = pd.read_csv(
        REFERENCE_FILE
    )

    current_df = pd.read_csv(
        batch_file
    )

    # ========================================================
    # O2 - DATA DRIFT
    # ========================================================

    data_report = monitor_numeric_features(

        reference_df,

        current_df,

        NUMERIC_COLUMNS
    )

    print("\nO2 - DATA DRIFT MONITORING")

    print(
        data_report.to_string(
            index=False
        )
    )

    # ========================================================
    # MAX PSI
    # ========================================================

    valid_psi = (
        data_report["psi"]
        .dropna()
    )

    if valid_psi.empty:

        max_psi = 0.0

    else:

        max_psi = float(
            valid_psi.max()
        )

    print(
        f"\nMAX PSI: {max_psi:.4f}"
    )

    # ========================================================
    # O3 - MODEL MONITORING
    # ========================================================

    model_metrics = (
        evaluate_production_batch(
            batch_file
        )
    )

    model_report = pd.DataFrame(
        [model_metrics]
    )

    print("\nO3 - MODEL MONITORING")

    print(
        model_report.to_string(
            index=False
        )
    )

    current_f1 = float(
        model_metrics["f1"]
    )

    # ========================================================
    # O4 - RETRAINING DECISION
    # ========================================================

    decision = should_retrain(

        max_psi=max_psi,

        current_f1=current_f1
    )

    print(
        "\nO4 - RETRAINING DECISION"
    )

    print(
        f"Max PSI     : {max_psi:.4f}"
    )

    print(
        f"Current F1  : {current_f1:.4f}"
    )

    print(
        f"Retrain     : "
        f"{decision['retrain']}"
    )

    print(
        f"Reason      : "
        f"{decision['reason']}"
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    result = {

        "batch_file":
            str(batch_file),

        "system_report":
            system_report,

        "data_report":
            data_report,

        "model_report":
            model_report,

        "max_psi":
            max_psi,

        "current_f1":
            current_f1,

        "retrain":
            decision["retrain"],

        "reason":
            decision["reason"]
    }

    print("\n" + "=" * 80)
    print("MONITOREO COMPLETO")
    print("=" * 80)

    return result


def main():

    run_monitoring()


if __name__ == "__main__":

    main()
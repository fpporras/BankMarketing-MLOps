from pathlib import Path

import pandas as pd

from src.monitoring.data_monitor import monitor_numeric_features
from src.monitoring.model_monitor import monitor_model
from src.monitoring.system_monitor import monitor_system


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bank_marketing.csv"
)


def main():

    print("\n" + "=" * 70)
    print("MONITORING - BANK MARKETING")
    print("=" * 70)

    # ========================================================
    # O1 - SYSTEM MONITORING
    # ========================================================

    system_report = monitor_system()

    print("\nO1 - SYSTEM MONITORING")
    print(system_report.to_string(index=False))

    # ========================================================
    # O2 - DATA MONITORING
    # ========================================================

    df = pd.read_csv(DATA_PATH)

    # División temporal de validación.
    # Cuando existan datos reales de producción,
    # current_df deberá reemplazarse por esos datos.
    split_index = int(len(df) * 0.70)

    reference_df = df.iloc[:split_index].copy()
    current_df = df.iloc[split_index:].copy()

    numeric_columns = [
        "age",
        "balance",
        "campaign",
        "pdays",
        "previous",
    ]

    data_report = monitor_numeric_features(
        reference_df,
        current_df,
        numeric_columns,
    )

    print("\nO2 - DATA MONITORING")
    print(data_report.to_string(index=False))

    # ========================================================
    # O3 - MODEL MONITORING
    # ========================================================

    model_report = monitor_model()

    print("\nO3 - MODEL MONITORING")
    print(model_report.to_string(index=False))

    # ========================================================
    # FIN
    # ========================================================

    print("\n" + "=" * 70)
    print("MONITORING COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
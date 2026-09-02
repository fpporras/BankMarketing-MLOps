from pathlib import Path
import pandas as pd

from src.monitoring.data_monitor import monitor_numeric_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bank_marketing.csv"
)


def main():
    df = pd.read_csv(DATA_PATH)

    split_index = int(len(df) * 0.70)

    reference_df = df.iloc[:split_index].copy()
    current_df = df.iloc[split_index:].copy()

    numeric_columns = [
        "age",
        "balance",
        "campaign",
        "pdays",
        "previous"
    ]

    report = monitor_numeric_features(
        reference_df,
        current_df,
        numeric_columns
    )

    print("\nO2 - DATA MONITORING")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
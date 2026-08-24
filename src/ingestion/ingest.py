from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def ingest_bank_marketing():
    """
    Download the Bank Marketing dataset from the UCI Machine Learning Repository
    and save the raw dataset locally.
    """

    print("Starting Bank Marketing dataset ingestion...")

    # Create raw data directory if it does not exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch dataset from UCI
    bank_marketing = fetch_ucirepo(id=222)

    # Separate features and target
    X = bank_marketing.data.features
    y = bank_marketing.data.targets

    # Combine features and target into one DataFrame
    df_raw = pd.concat([X, y], axis=1)

    # Save raw dataset
    output_file = RAW_DATA_DIR / "bank_marketing.csv"
    df_raw.to_csv(output_file, index=False)

    print(f"Dataset saved to: {output_file}")
    print(f"Rows: {df_raw.shape[0]}")
    print(f"Columns: {df_raw.shape[1]}")

    print("\nColumns:")
    print(df_raw.columns.tolist())

    print("\nTarget distribution:")
    print(df_raw["y"].value_counts())

    return df_raw


if __name__ == "__main__":
    ingest_bank_marketing()
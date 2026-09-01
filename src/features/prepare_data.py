from pathlib import Path
from src.features.build_features import build_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)

PROCESSED_FILE = (
    PROCESSED_DIR /
    "df_features.csv"
)


def prepare_processed_data(df_raw):

    print("=" * 70)
    print("ETAPA — FEATURE ENGINEERING")
    print("=" * 70)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Dataset RAW: {df_raw.shape}"
    )

    df_processed = build_features(
        df_raw
    )

    print(
        f"Dataset PROCESSED: {df_processed.shape}"
    )

    df_processed.to_csv(
        PROCESSED_FILE,
        index=False
    )

    print(
        f"Dataset procesado guardado en:\n"
        f"{PROCESSED_FILE}"
    )

    return df_processed
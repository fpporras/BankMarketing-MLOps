from pathlib import Path

from sklearn.model_selection import train_test_split

from src.ingestion.ingest import load_raw_data
from src.features.build_features import build_features


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "reference"
)


REFERENCE_FILE = (
    REFERENCE_DIR
    / "reference.csv"
)


def create_reference():

    REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_raw_data()

    df = build_features(
        df
    )

    reference, _ = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["y"]
    )

    reference.to_csv(
        REFERENCE_FILE,
        index=False
    )

    print(
        f"Reference dataset creado: "
        f"{REFERENCE_FILE}"
    )


if __name__ == "__main__":

    create_reference()
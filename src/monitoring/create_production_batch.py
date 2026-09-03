from pathlib import Path

import pandas as pd

from src.ingestion.ingest import (
    load_raw_data
)

from src.features.build_features import (
    build_features
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

PRODUCTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "production"
)


def create_batch(
    batch_number,
    sample_size=1000
):

    PRODUCTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_raw_data()

    df = build_features(
        df
    )

    batch = df.sample(
        n=sample_size,
        random_state=
            100 + batch_number
    )

    output = (
        PRODUCTION_DIR
        / f"batch_{batch_number}.csv"
    )

    batch.to_csv(
        output,
        index=False
    )

    print(
        f"Batch creado: {output}"
    )


if __name__ == "__main__":

    for batch in range(1, 4):

        create_batch(
            batch
        )
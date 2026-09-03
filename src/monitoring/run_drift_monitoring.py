from pathlib import Path

import pandas as pd

from src.monitoring.data_drift import (
    calculate_psi,
    classify_psi
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


def monitor_batch(
    batch_number
):

    reference = pd.read_csv(
        REFERENCE_FILE
    )

    production = pd.read_csv(
        PRODUCTION_DIR
        / f"batch_{batch_number}.csv"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"DRIFT MONITORING - "
        f"BATCH {batch_number}"
    )

    print(
        "=" * 70
    )

    results = []

    for column in NUMERIC_COLUMNS:

        psi = calculate_psi(

            reference[column]
            .dropna(),

            production[column]
            .dropna()
        )

        status = classify_psi(
            psi
        )

        results.append({

            "feature":
                column,

            "psi":
                psi,

            "status":
                status
        })

        print(
            f"{column:15} "
            f"PSI={psi:.4f} "
            f"STATUS={status}"
        )

    return results


if __name__ == "__main__":

    for batch in [1, 2, 3]:

        monitor_batch(
            batch
        )
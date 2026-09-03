from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

PRODUCTION_DIR = (
    PROJECT_ROOT
    / "monitoring"
    / "production"
)


def simulate_drift():

    batch_path = (
        PRODUCTION_DIR
        / "batch_3.csv"
    )

    df = pd.read_csv(
        batch_path
    )

    # --------------------------------------------------------
    # Simulación de cambio de distribución
    # --------------------------------------------------------

    df["age"] = (
        df["age"] + 15
    ).clip(
        upper=100
    )

    df["balance"] = (
        df["balance"] * 2
    )

    df["campaign"] = (
        df["campaign"] + 3
    )

    df.to_csv(
        batch_path,
        index=False
    )

    print(
        "Drift simulado en batch_3."
    )


if __name__ == "__main__":

    simulate_drift()
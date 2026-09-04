from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

PRODUCTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "production"
)


def simulate_drift():

    batch_path = (
        PRODUCTION_DIR
        / "batch_3.csv"
    )

    if not batch_path.exists():

        raise FileNotFoundError(
            f"No existe: {batch_path}\n"
            "Primero crea los production batches."
        )

    df = pd.read_csv(
        batch_path
    )

    print("\nSimulando data drift...")

    # ========================================================
    # AGE DRIFT
    # ========================================================

    df["age"] = (
        df["age"] + 15
    ).clip(
        upper=100
    )

    # ========================================================
    # BALANCE DRIFT
    # ========================================================

    df["balance"] = (
        df["balance"] * 2
    )

    # ========================================================
    # CAMPAIGN DRIFT
    # ========================================================

    df["campaign"] = (
        df["campaign"] + 3
    )

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        batch_path,
        index=False
    )

    print(
        f"Drift simulado en:"
        f"\n{batch_path}"
    )


if __name__ == "__main__":

    simulate_drift()
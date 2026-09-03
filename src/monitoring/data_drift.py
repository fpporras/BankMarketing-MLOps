from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)


REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "reference"
)


PRODUCTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "production"
)


PSI_WARNING = 0.10

PSI_ALERT = 0.25


def calculate_psi(
    reference,
    production,
    bins=10
):

    reference = np.asarray(
        reference,
        dtype=float
    )

    production = np.asarray(
        production,
        dtype=float
    )

    breakpoints = np.unique(
        np.percentile(
            reference,
            np.linspace(
                0,
                100,
                bins + 1
            )
        )
    )

    if len(breakpoints) < 3:

        return 0.0

    reference_counts, _ = np.histogram(
        reference,
        bins=breakpoints
    )

    production_counts, _ = np.histogram(
        production,
        bins=breakpoints
    )

    reference_pct = (
        reference_counts
        / len(reference)
    )

    production_pct = (
        production_counts
        / len(production)
    )

    reference_pct = np.clip(
        reference_pct,
        0.0001,
        None
    )

    production_pct = np.clip(
        production_pct,
        0.0001,
        None
    )

    psi = np.sum(

        (
            production_pct
            - reference_pct
        )

        *

        np.log(
            production_pct
            / reference_pct
        )
    )

    return float(psi)


def classify_psi(psi):

    if psi < PSI_WARNING:

        return "OK"

    if psi < PSI_ALERT:

        return "WARNING"

    return "ALERT"
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank_marketing.csv"

REQUIRED_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day_of_week",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y",
]


def load_data():
    return pd.read_csv(DATA_PATH)


def test_dataset_exists():
    assert DATA_PATH.exists(), f"No se encontró el dataset en: {DATA_PATH}"


def test_required_columns():
    df = load_data()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    assert not missing_columns, (
        f"Faltan columnas obligatorias: {missing_columns}"
    )


def test_target_has_no_missing_values():
    df = load_data()

    assert df["y"].isna().sum() == 0, (
        "La variable objetivo 'y' contiene valores faltantes"
    )


def test_target_has_valid_categories():
    df = load_data()

    valid_values = {"yes", "no"}
    observed_values = set(df["y"].dropna().unique())

    assert observed_values.issubset(valid_values), (
        f"Se encontraron categorías inválidas en 'y': "
        f"{observed_values - valid_values}"
    )


def test_age_is_numeric():
    df = load_data()

    assert pd.api.types.is_numeric_dtype(df["age"]), (
        "La variable 'age' debe ser numérica"
    )


def test_age_valid_range():
    df = load_data()

    assert df["age"].between(18, 100).all(), (
        "Se encontraron valores de 'age' fuera del rango esperado"
    )


def test_dataset_not_empty():
    df = load_data()

    assert len(df) > 0, "El dataset está vacío"
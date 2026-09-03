from src.ingestion.ingest import load_raw_data
from src.validation.quality_gates import (
    EXPECTED_COLUMNS,
)


def test_dataset_not_empty():

    df = load_raw_data()

    assert len(df) > 0


def test_schema():

    df = load_raw_data()

    assert set(
        df.columns
    ) == set(
        EXPECTED_COLUMNS
    )


def test_target_exists():

    df = load_raw_data()

    assert "y" in df.columns


def test_target_values():

    df = load_raw_data()

    assert set(
        df["y"].unique()
    ).issubset(
        {"yes", "no"}
    )


def test_age_range():

    df = load_raw_data()

    assert (
        df["age"]
        .between(18, 100)
        .all()
    )
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# Rutas del proyecto y del archivo de datos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def ingest_bank_marketing():
    """
    Descarga el dataset de Bank Marketing desde UCI Machine Learning Repository y guarda el dataset raw localmente.
    """

    print("Iniciando la ingesta del conjunto de datos de Bank Marketing...")

    # Crea el directorio de datos raw si no existe
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Trae el dataset desde UCI
    bank_marketing = fetch_ucirepo(id=222)

    # Separa las features y target (X y Y)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets

    # Combina las features (X) y target (Y) en un mismo DataFrame
    df_raw = pd.concat([X, y], axis=1)

    # Guarda el dataset raw
    output_file = RAW_DATA_DIR / "bank_marketing.csv"
    df_raw.to_csv(output_file, index=False)

    print(f"Dataset guardado en: {output_file}")
    print(f"Filas: {df_raw.shape[0]}")
    print(f"Columnas: {df_raw.shape[1]}")

    print("\nColumnas:")
    print(df_raw.columns.tolist())

    print("\nDistribución del target:")
    print(df_raw["y"].value_counts())

    return df_raw


if __name__ == "__main__":
    ingest_bank_marketing()
from pathlib import Path

import pandas as pd


# Rutas del proyecto y del archivo de datos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "bank_marketing.csv"


def load_raw_data():
    """Carga el dataset de Bank Marketing desde el archivo CSV raw localmente."""

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {RAW_DATA_FILE}"
        )

    df_raw = pd.read_csv(RAW_DATA_FILE)

    return df_raw


def run_data_quality_diagnosis(df_raw):
    """Realiza un diagnóstico básico de la calidad de los datos."""

    print("=" * 60)
    print("DIAGNÓSTICO DE CALIDAD DE LOS DATOS")
    print("=" * 60)

    print("\n1. Dimensiones del dataset")
    print(f"Filas: {df_raw.shape[0]}")
    print(f"Columnas: {df_raw.shape[1]}")

    print("\n2. Nombres de las columnas")
    print(df_raw.columns.tolist())

    print("\n3. Tipos de datos")
    print(df_raw.dtypes)

    print("\n4. Valores faltantes")
    missing_values = df_raw.isnull().sum()
    print(missing_values[missing_values > 0])

    print("\n5. Porcentaje de valores faltantes")
    missing_percentage = (df_raw.isnull().mean() * 100)
    print(missing_percentage[missing_percentage > 0])

    print("\n6. Filas duplicadas")
    duplicate_count = df_raw.duplicated().sum()
    duplicate_percentage = (
        duplicate_count / len(df_raw) * 100
    )

    print(f"Filas duplicadas: {duplicate_count}")
    print(f"Porcentaje de filas duplicadas: {duplicate_percentage:.2f}%")

    print("\n7. Distribución del target")
    print(df_raw["y"].value_counts())

    print("\n8. Distribución del target (porcentaje)")
    print(
        df_raw["y"].value_counts(normalize=True).mul(100).round(2)
    )

    print("\n9. Valores únicos por columna")

    cardinality = df_raw.nunique().sort_values()

    print(cardinality)

    print("\n10. Resumen numérico")

    print(df_raw.describe().T)

    print("\n11. Resumen categórico")

    categorical_columns = df_raw.select_dtypes(
        include=["object"]
    ).columns

    print(df_raw[categorical_columns].describe().T)

    print("\n12. 'unknown' values")

    unknown_counts = {}

    for column in df_raw.columns:
        count = (df_raw[column] == "unknown").sum()

        if count > 0:
            unknown_counts[column] = count

    if unknown_counts:
        for column, count in unknown_counts.items():
            percentage = count / len(df_raw) * 100
            print(
                f"{column}: {count} ({percentage:.2f}%)"
            )
    else:
        print("No se encontraron valores 'unknown'.")


if __name__ == "__main__":

    df_raw = load_raw_data()

    run_data_quality_diagnosis(df_raw)
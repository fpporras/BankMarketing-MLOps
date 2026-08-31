from pathlib import Path

from src.ingestion.ingest import load_raw_data
from src.features.build_features import build_features


# ============================================================
# RUTAS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_FILE = (
    PROCESSED_DIR
    / "df_features.csv"
)


# ============================================================
# PREPARACIÓN
# ============================================================

def prepare_processed_data():

    print("=" * 70)
    print("PREPARANDO DATASET PROCESADO")
    print("=" * 70)

    # --------------------------------------------------------
    # Crear directorio
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Cargar RAW
    # --------------------------------------------------------

    df_raw = load_raw_data()

    print(
        f"Dataset RAW: {df_raw.shape}"
    )

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    df_processed = build_features(
        df_raw
    )

    print(
        f"Dataset PROCESSED: "
        f"{df_processed.shape}"
    )

    # --------------------------------------------------------
    # Guardar
    # --------------------------------------------------------

    df_processed.to_csv(
        PROCESSED_FILE,
        index=False
    )

    print(
        f"\nDataset procesado guardado en:\n"
        f"{PROCESSED_FILE}"
    )

    return df_processed


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    prepare_processed_data()
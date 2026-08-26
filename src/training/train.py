from sklearn.model_selection import train_test_split

from src.ingestion.ingest import load_raw_data
from src.validation.quality_gates import run_quality_gates
from src.features.build_features import build_features
from src.preprocessing.preprocess import build_preprocessor


def main():

    # ========================================================
    # 1. INGESTA
    # ========================================================

    df_raw = load_raw_data()

    print("Datos cargados correctamente.")
    print(f"Dimensiones originales: {df_raw.shape}")

    # ========================================================
    # 2. QUALITY GATES
    # ========================================================

    quality_passed = run_quality_gates(df_raw)

    if not quality_passed:
        raise SystemExit(
            "El pipeline fue bloqueado por calidad de datos."
        )

    # ========================================================
    # 3. FEATURE ENGINEERING
    # ========================================================

    df_features = build_features(df_raw)

    print("\nFeature Engineering completado.")
    print(f"Dimensiones finales: {df_features.shape}")

    print("\nColumnas:")
    print(df_features.columns.tolist())

    print("\nPrimeras filas:")
    print(df_features.head())

    # ========================================================
    # 4. SEPARAR X E y
    # ========================================================

    X = df_features.drop(
        columns=["y"]
    )

    y = df_features["y"]

    print("\nSeparación X / y completada.")
    print(f"Dimensiones de X: {X.shape}")
    print(f"Dimensiones de y: {y.shape}")

    # ========================================================
    # 5. TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTrain/Test Split completado.")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    # ========================================================
    # 6. CONSTRUIR PREPROCESADOR
    # ========================================================

    preprocessor = build_preprocessor(
        X_train
    )

    print("\nPreprocesador construido correctamente.")

    # ========================================================
    # 7. APLICAR PREPROCESAMIENTO
    # ========================================================

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    print("\nPreprocesamiento aplicado correctamente.")
    print(f"X_train procesado: {X_train_processed.shape}")
    print(f"X_test procesado: {X_test_processed.shape}")


if __name__ == "__main__":
    main()
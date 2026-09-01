from pathlib import Path
import sys
import traceback

import mlflow

from src.ingestion.ingest import ingest_bank_marketing
from src.validation.data_quality import (
    run_data_quality_diagnosis,
    analyze_missing_values,
)
from src.validation.quality_gates import run_quality_gates
from src.features.prepare_data import prepare_processed_data
from src.training.train import train_models
from src.evaluation.promote_model import (
    promote_registered_model,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

MLFLOW_TRACKING_URI = (
    "http://127.0.0.1:5000"
)

REGISTERED_MODEL_NAME = (
    "bank-marketing-classifier"
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]  

MODELS_DIR = (
    PROJECT_ROOT /
    "models"
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOGGING
# ============================================================

def print_stage(number, title):

    print("\n")
    print("=" * 80)
    print(
        f"ETAPA {number} — {title}"
    )
    print("=" * 80)


# ============================================================
# ORCHESTRATOR
# ============================================================

def main():

    try:

        print("\n")
        print("=" * 80)
        print("BANK MARKETING ML PIPELINE")
        print("=" * 80)

        print(
            "\nPipeline iniciado correctamente."
        )

        # ====================================================
        # 1. INGESTION
        # ====================================================

        print_stage(
            1,
            "INGESTA"
        )

        df_raw = ingest_bank_marketing()

        print(
            "\n✓ Ingesta completada."
        )

        # ====================================================
        # 2. DATA QUALITY DIAGNOSIS
        # ====================================================

        print_stage(
            2,
            "DIAGNÓSTICO DE CALIDAD"
        )

        run_data_quality_diagnosis(
            df_raw
        )

        analyze_missing_values(
            df_raw
        )

        print(
            "\n✓ Diagnóstico completado."
        )

        # ====================================================
        # 3. QUALITY GATES
        # ====================================================

        print_stage(
            3,
            "QUALITY GATES"
        )

        quality_passed = (
            run_quality_gates(
                df_raw
            )
        )

        if not quality_passed:

            print(
                "\n✗ QUALITY GATES FALLARON."
            )

            print(
                "El pipeline ha sido bloqueado."
            )

            return 1

        print(
            "\n✓ QUALITY GATES APROBADOS."
        )

        # ====================================================
        # 4. FEATURE ENGINEERING
        # ====================================================

        print_stage(
            4,
            "FEATURE ENGINEERING"
        )

        df_features = (
            prepare_processed_data(
                df_raw
            )
        )

        print(
            "\n✓ Feature engineering completado."
        )

        # ====================================================
        # 5. TRAINING
        # ====================================================

        print_stage(
            5,
            "TRAINING"
        )

        training_results = train_models(
            df_features
        )

        print(
            "\n✓ Entrenamiento completado."
        )

        # ====================================================
        # 6. RESULTADO DEL MODELO
        # ====================================================

        best_model_name = (
            training_results[
                "best_model_name"
            ]
        )

        best_model = (
            training_results[
                "best_model"
            ]
        )

        best_metrics = (
            training_results[
                "best_metrics"
            ]
        )

        best_grid = (
            training_results[
                "best_grid"
            ]
        )

        algorithm_mapping = (
            training_results[
                "algorithm_mapping"
            ]
        )

        feature_set = (
            training_results[
                "feature_set"
            ]
        )

        print(
            "\n" +
            "=" * 80
        )

        print(
            "MEJOR MODELO"
        )

        print(
            "=" * 80
        )

        print(
            f"Modelo: {best_model_name}"
        )

        print(
            f"F1: {best_metrics['f1']:.4f}"
        )

        print(
            f"Recall: {best_metrics['recall']:.4f}"
        )

        print(
            f"ROC-AUC: {best_metrics['roc_auc']:.4f}"
        )

        # ====================================================
        # 7. MLFLOW
        # ====================================================

        print_stage(
            6,
            "MLFLOW"
        )

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        print(
            "✓ MLflow configurado."
        )

        # ====================================================
        # 8. REGISTER MODEL
        # ====================================================

        print_stage(
            7,
            "MODEL REGISTRATION"
        )

        registered = (
            training_results[
                "registered"
            ]
        )

        if not registered:

            training_results[
                "register_model"
            ](
                best_model_name
            )

        print(
            "\n✓ Modelo registrado en MLflow."
        )

        # ====================================================
        # 9. PROMOTION
        # ====================================================

        print_stage(
            8,
            "MODEL VALIDATION & PROMOTION"
        )

        promoted = (
            promote_registered_model(
                model_name=REGISTERED_MODEL_NAME,
                metrics=best_metrics
            )
        )

        if promoted:

            print(
                "\n✓ Modelo promocionado a champion."
            )

        else:

            print(
                "\n⚠ Modelo no promocionado."
            )

        # ====================================================
        # 10. SAVE MODEL
        # ====================================================

        print_stage(
            9,
            "MODEL ARTIFACT"
        )

        model_path = (
            MODELS_DIR /
            "best_model.joblib"
        )

        training_results[
            "save_model"
        ](
            best_model,
            model_path
        )

        print(
            f"\n✓ Modelo guardado en:"
            f"\n{model_path}"
        )

        # ====================================================
        # FIN
        # ====================================================

        print("\n")
        print("=" * 80)
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 80)

        return 0

    except Exception as error:

        print("\n")
        print("=" * 80)
        print("PIPELINE FAILED")
        print("=" * 80)

        print(
            f"\nError: {error}"
        )

        traceback.print_exc()

        return 1


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )
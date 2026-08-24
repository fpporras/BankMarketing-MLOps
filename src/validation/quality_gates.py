from pathlib import Path

import pandas as pd


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bank_marketing.csv"
)


# ============================================================
# CONFIGURACIÓN DEL DATASET
# ============================================================

EXPECTED_COLUMNS = [
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

VALID_TARGET_VALUES = {"yes", "no"}

DUPLICATE_RATE_THRESHOLD = 0.01

MINIMUM_ROWS = 1000

MINIMUM_AGE = 18
MAXIMUM_AGE = 100


# ============================================================
# DEFINICIONES DE SEVERIDAD
# ============================================================

SEVERIDAD_INFORMACION = "INFORMACIÓN"
SEVERIDAD_ADVERTENCIA = "ADVERTENCIA"
SEVERIDAD_ERROR = "ERROR"
SEVERIDAD_CRITICO = "CRÍTICO"


# ============================================================
# CARGA DEL DATASET
# ============================================================

def load_raw_data():
    """Carga el dataset raw de Bank Marketing."""

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {RAW_DATA_FILE}"
        )

    df_raw = pd.read_csv(RAW_DATA_FILE)

    return df_raw


# ============================================================
# RESULTADO ESTANDARIZADO
# ============================================================

def create_result(nombre, estado, severidad, mensaje):
    """
    Crea un resultado estandarizado para una puerta de calidad.
    """

    return {
        "Nombre": nombre,
        "Estado": estado,
        "Severidad": severidad,
        "Mensaje": mensaje,
    }


# ============================================================
# GATE 1 — DATASET NO VACÍO
# ============================================================

def check_dataset_not_empty(df_raw):
    """Comprueba que el dataset contenga registros."""

    if len(df_raw) > 0:

        return create_result(
            "El dataset no está vacío",
            "PASA",
            SEVERIDAD_INFORMACION,
            f"El dataset contiene {len(df_raw)} filas.",
        )

    return create_result(
        "El dataset está vacío",
        "FALLO",
        SEVERIDAD_CRITICO,
        "El dataset no contiene registros.",
    )


# ============================================================
# GATE 2 — CANTIDAD MÍNIMA DE REGISTROS
# ============================================================

def check_minimum_rows(df_raw):
    """Comprueba que el dataset tenga suficientes registros."""

    row_count = len(df_raw)

    if row_count >= MINIMUM_ROWS:

        return create_result(
            "Cantidad mínima de registros",
            "PASA",
            SEVERIDAD_INFORMACION,
            (
                f"El dataset contiene {row_count} registros, "
                f"superando el mínimo requerido de "
                f"{MINIMUM_ROWS}."
            ),
        )

    return create_result(
        "Cantidad mínima de registros",
        "FALLO",
        SEVERIDAD_CRITICO,
        (
            f"El dataset contiene solamente {row_count} registros. "
            f"Se requieren al menos {MINIMUM_ROWS}."
        ),
    )


# ============================================================
# GATE 3 — ESQUEMA
# ============================================================

def check_schema(df_raw):
    """Comprueba que el esquema sea el esperado."""

    actual_columns = set(df_raw.columns)
    expected_columns = set(EXPECTED_COLUMNS)

    missing_columns = expected_columns - actual_columns
    unexpected_columns = actual_columns - expected_columns

    if not missing_columns and not unexpected_columns:

        return create_result(
            "Esquema del dataset",
            "PASA",
            SEVERIDAD_INFORMACION,
            "Todas las columnas esperadas están presentes.",
        )

    return create_result(
        "Esquema del dataset",
        "FALLO",
        SEVERIDAD_CRITICO,
        (
            f"Columnas faltantes: {sorted(missing_columns)}. "
            f"Columnas inesperadas: "
            f"{sorted(unexpected_columns)}."
        ),
    )


# ============================================================
# GATE 4 — TARGET SIN VALORES FALTANTES
# ============================================================

def check_target_missing(df_raw):
    """Comprueba que el target no tenga valores faltantes."""

    missing_count = df_raw["y"].isna().sum()

    if missing_count == 0:

        return create_result(
            "Valores faltantes en el target",
            "PASA",
            SEVERIDAD_INFORMACION,
            "El target no contiene valores faltantes.",
        )

    return create_result(
        "Valores faltantes en el target",
        "FALLO",
        SEVERIDAD_CRITICO,
        (
            f"El target contiene {missing_count} "
            f"valores faltantes."
        ),
    )


# ============================================================
# GATE 5 — VALORES VÁLIDOS DEL TARGET
# ============================================================

def check_target_values(df_raw):
    """Comprueba que los valores del target sean válidos."""

    actual_values = set(
        df_raw["y"].dropna().unique()
    )

    invalid_values = (
        actual_values - VALID_TARGET_VALUES
    )

    if not invalid_values:

        return create_result(
            "Categorías del target",
            "PASA",
            SEVERIDAD_INFORMACION,
            (
                "Las categorías del target son válidas: "
                f"{sorted(actual_values)}."
            ),
        )

    return create_result(
        "Categorías del target",
        "FALLO",
        SEVERIDAD_CRITICO,
        (
            "Se encontraron valores inválidos en el target: "
            f"{sorted(invalid_values)}."
        ),
    )


# ============================================================
# GATE 6 — RANGO DE EDAD
# ============================================================

def check_age_range(df_raw):
    """Comprueba que la edad esté dentro de un rango razonable."""

    invalid_count = (
        (df_raw["age"] < MINIMUM_AGE)
        | (df_raw["age"] > MAXIMUM_AGE)
    ).sum()

    if invalid_count == 0:

        return create_result(
            "Rango de edad",
            "PASA",
            SEVERIDAD_INFORMACION,
            (
                f"Todas las edades están entre "
                f"{MINIMUM_AGE} y {MAXIMUM_AGE} años."
            ),
        )

    return create_result(
        "Rango de edad",
        "FALLO",
        SEVERIDAD_ERROR,
        (
            f"Se encontraron {invalid_count} "
            f"registros con una edad fuera del rango "
            f"{MINIMUM_AGE}-{MAXIMUM_AGE}."
        ),
    )


# ============================================================
# GATE 7 — DUPLICADOS
# ============================================================

def check_duplicate_rate(df_raw):
    """Comprueba la tasa de registros duplicados."""

    duplicate_count = df_raw.duplicated().sum()

    duplicate_rate = (
        duplicate_count / len(df_raw)
    )

    if duplicate_rate <= DUPLICATE_RATE_THRESHOLD:

        return create_result(
            "Tasa de registros duplicados",
            "PASA",
            SEVERIDAD_INFORMACION,
            (
                f"La tasa de duplicados es "
                f"{duplicate_rate:.2%}, "
                f"por debajo del umbral de "
                f"{DUPLICATE_RATE_THRESHOLD:.2%}."
            ),
        )

    return create_result(
        "Tasa de registros duplicados",
        "FALLO",
        SEVERIDAD_ERROR,
        (
            f"La tasa de duplicados es "
            f"{duplicate_rate:.2%}, "
            f"superando el umbral permitido de "
            f"{DUPLICATE_RATE_THRESHOLD:.2%}."
        ),
    )


# ============================================================
# GATE 8 — VALORES FALTANTES EN VARIABLES
# ============================================================

def check_missing_values(df_raw):
    """
    Comprueba la cantidad de valores faltantes
    en las variables predictoras.
    """

    missing_percentage = (
        df_raw.isna().mean() * 100
    )

    columns_with_missing = (
        missing_percentage[
            missing_percentage > 0
        ]
    )

    if columns_with_missing.empty:

        return create_result(
            "Valores faltantes en variables",
            "PASA",
            SEVERIDAD_INFORMACION,
            "No se encontraron valores faltantes.",
        )

    details = []

    for column, percentage in columns_with_missing.items():

        details.append(
            f"{column}: {percentage:.2f}%"
        )

    return create_result(
        "Valores faltantes en variables",
        "ADVERTENCIA",
        SEVERIDAD_ADVERTENCIA,
        (
            "Se encontraron valores faltantes: "
            + ", ".join(details)
            + ". Se requiere tratamiento antes del entrenamiento."
        ),
    )


# ============================================================
# EJECUCIÓN DE QUALITY GATES
# ============================================================

def run_quality_gates(df_raw):
    """Ejecuta todas las puertas de calidad."""

    gates = [

        check_dataset_not_empty(df_raw),

        check_minimum_rows(df_raw),

        check_schema(df_raw),

        check_target_missing(df_raw),

        check_target_values(df_raw),

        check_age_range(df_raw),

        check_duplicate_rate(df_raw),

        check_missing_values(df_raw),

    ]

    print("=" * 70)
    print("PUERTAS DE CALIDAD DE LOS DATOS")
    print("=" * 70)

    for gate in gates:

        if gate["Estado"] == "PASA":
            simbolo = "✓"

        else:
            simbolo = "✗"

        print(
            f"[{simbolo}] "
            f"{gate['Nombre']}"
        )

        print(
            f"    Estado: {gate['Estado']}"
        )

        print(
            f"    Severidad: {gate['Severidad']}"
        )

        print(
            f"    Mensaje: {gate['Mensaje']}"
        )

        print()

    # --------------------------------------------------------
    # Identificar fallos críticos
    # --------------------------------------------------------

    critical_failures = [
        gate
        for gate in gates
        if (
            gate["Estado"] == "FALLO"
            and gate["Severidad"]
            == SEVERIDAD_CRITICO
        )
    ]

    # --------------------------------------------------------
    # Identificar errores
    # --------------------------------------------------------

    errors = [
        gate
        for gate in gates
        if (
            gate["Estado"] == "FALLO"
            and gate["Severidad"]
            == SEVERIDAD_ERROR
        )
    ]

    # --------------------------------------------------------
    # Identificar advertencias
    # --------------------------------------------------------

    warnings = [
        gate
        for gate in gates
        if gate["Severidad"]
        == SEVERIDAD_ADVERTENCIA
    ]

    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    print("=" * 70)
    print("RESUMEN DE CALIDAD")
    print("=" * 70)

    print(
        f"Total de validaciones: {len(gates)}"
    )

    print(
        f"Validaciones aprobadas: "
        f"{sum(gate['Estado'] == 'PASA' for gate in gates)}"
    )

    print(
        f"Fallos críticos: "
        f"{len(critical_failures)}"
    )

    print(
        f"Errores: "
        f"{len(errors)}"
    )

    print(
        f"Advertencias: "
        f"{len(warnings)}"
    )

    print("-" * 70)

    # Un CRÍTICO bloquea el pipeline
    if critical_failures:

        print(
            "RESULTADO FINAL: BLOQUEADO"
        )

        print(
            "Existen fallos críticos de calidad."
        )

        return False

    # ERROR no bloquea todavía,
    # pero requiere tratamiento.
    if errors:

        print(
            "RESULTADO FINAL: APROBADO CON ERRORES"
        )

        print(
            "Existen errores que deben "
            "ser tratados antes del entrenamiento."
        )

        return True

    # WARNING permite continuar.
    if warnings:

        print(
            "RESULTADO FINAL: APROBADO CON ADVERTENCIAS"
        )

        return True

    print(
        "RESULTADO FINAL: APROBADO"
    )

    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df_raw = load_raw_data()

    quality_passed = run_quality_gates(
        df_raw
    )

    if not quality_passed:

        raise SystemExit(
            "Las puertas de calidad fallaron. "
            "El pipeline ha sido bloqueado."
        )
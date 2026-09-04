import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN DE UMBRALES PSI
# ============================================================

# Umbrales heurísticos usados como criterios de monitoreo:
# PSI < 0.10        -> cambio pequeño / estable
# 0.10 <= PSI < 0.25 -> cambio moderado / advertencia
# PSI >= 0.25       -> cambio importante / drift
#
# Estos valores funcionan como reglas de alerta,
# no como pruebas estadísticas absolutas.

PSI_WARNING_THRESHOLD = 0.10
PSI_DRIFT_THRESHOLD = 0.25


# ============================================================
# CLASIFICACIÓN DEL PSI
# ============================================================

def classify_psi(psi_value):
    """
    Clasifica el valor PSI en tres niveles:
    OK, WARNING o DRIFT.
    """

    if psi_value < PSI_WARNING_THRESHOLD:
        return "OK"

    elif psi_value < PSI_DRIFT_THRESHOLD:
        return "WARNING"

    else:
        return "DRIFT"


# ============================================================
# CÁLCULO DEL PSI PARA VARIABLES NUMÉRICAS
# ============================================================

def calculate_numeric_psi(reference, current, bins=10):
    """
    Calcula el Population Stability Index (PSI)
    entre una distribución de referencia y una distribución actual.

    Parameters
    ----------
    reference : array-like
        Datos de referencia.

    current : array-like
        Datos actuales o de producción.

    bins : int
        Cantidad de intervalos utilizados para comparar distribuciones.

    Returns
    -------
    float
        Valor PSI.
    """

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    # Evitar errores si alguna serie queda vacía
    if reference.empty or current.empty:
        return np.nan

    # Si la variable de referencia es constante,
    # no se pueden generar intervalos útiles.
    if reference.nunique() <= 1:
        if current.nunique() <= 1:
            return 0.0
        return np.inf

    # Los puntos de corte se calculan SOLO
    # a partir de los datos de referencia.
    breakpoints = np.percentile(
        reference,
        np.linspace(0, 100, bins + 1)
    )

    # Eliminar puntos repetidos
    breakpoints = np.unique(breakpoints)

    # Si quedan menos de dos cortes,
    # no es posible construir un histograma.
    if len(breakpoints) < 2:
        return 0.0

    # Permitir valores actuales más pequeños o más grandes
    # que los observados en referencia.
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Frecuencias de referencia
    reference_counts, _ = np.histogram(
        reference,
        bins=breakpoints
    )

    # Frecuencias actuales
    current_counts, _ = np.histogram(
        current,
        bins=breakpoints
    )

    # Convertir frecuencias en proporciones
    reference_pct = (
        reference_counts / reference_counts.sum()
    )

    current_pct = (
        current_counts / current_counts.sum()
    )

    # Evitar divisiones entre cero y log(0)
    epsilon = 0.0001

    reference_pct = np.where(
        reference_pct == 0,
        epsilon,
        reference_pct
    )

    current_pct = np.where(
        current_pct == 0,
        epsilon,
        current_pct
    )

    # Fórmula PSI
    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


# ============================================================
# MONITOREO DE VARIAS VARIABLES NUMÉRICAS
# ============================================================

def monitor_numeric_features(reference_df, current_df, columns):
    """
    Calcula PSI para varias columnas numéricas
    y devuelve un DataFrame con los resultados.
    """

    results = []

    for column in columns:

        # Validar que la columna exista en ambos datasets
        if column not in reference_df.columns:
            raise KeyError(
                f"La columna '{column}' no existe en reference_df."
            )

        if column not in current_df.columns:
            raise KeyError(
                f"La columna '{column}' no existe en current_df."
            )

        psi = calculate_numeric_psi(
            reference_df[column],
            current_df[column]
        )

        # Si PSI no es finito, requiere revisión manual
        if np.isnan(psi) or np.isinf(psi):
            status = "REVIEW"
        else:
            status = classify_psi(psi)

        results.append(
            {
                "variable": column,
                "psi": round(psi, 4)
                if np.isfinite(psi)
                else psi,
                "status": status
            }
        )

    return pd.DataFrame(results)
PSI_ALERT_THRESHOLD = 0.25

F1_MINIMUM_THRESHOLD = 0.50


def should_retrain(
    max_psi,
    current_f1
):
    """
    Determina si debe ejecutarse retraining.

    Reglas:

    1. Drift + degradación:
       retrain = True

    2. Solo drift:
       retrain = False

    3. Solo degradación:
       retrain = True

    4. Ninguna:
       retrain = False
    """

    drift_detected = (
        max_psi
        >= PSI_ALERT_THRESHOLD
    )

    performance_degraded = (
        current_f1
        < F1_MINIMUM_THRESHOLD
    )

    # --------------------------------------------------------
    # Drift + performance degradation
    # --------------------------------------------------------

    if (
        drift_detected
        and performance_degraded
    ):

        return {

            "retrain": True,

            "reason":
                "Data drift significativo "
                "y degradación del modelo."
        }

    # --------------------------------------------------------
    # Only drift
    # --------------------------------------------------------

    if drift_detected:

        return {

            "retrain": False,

            "reason":
                "Existe data drift, "
                "pero no se observa degradación "
                "suficiente del modelo."
        }

    # --------------------------------------------------------
    # Only performance degradation
    # --------------------------------------------------------

    if performance_degraded:

        return {

            "retrain": True,

            "reason":
                "El rendimiento del modelo "
                "está por debajo del umbral."
        }

    # --------------------------------------------------------
    # No problem
    # --------------------------------------------------------

    return {

        "retrain": False,

        "reason":
            "No se detectaron condiciones "
            "para reentrenamiento."
    }
PSI_ALERT_THRESHOLD = 0.25

F1_MINIMUM_THRESHOLD = 0.50


def should_retrain(
    max_psi,
    current_f1
):

    drift_detected = (
        max_psi
        >= PSI_ALERT_THRESHOLD
    )

    performance_degraded = (
        current_f1
        < F1_MINIMUM_THRESHOLD
    )

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

    if drift_detected:

        return {

            "retrain": False,

            "reason":
                "Existe data drift, "
                "pero no se observa degradación "
                "suficiente del modelo."
        }

    if performance_degraded:

        return {

            "retrain": True,

            "reason":
                "El rendimiento del modelo "
                "está por debajo del umbral."
        }

    return {

        "retrain": False,

        "reason":
            "No se detectaron condiciones "
            "para reentrenamiento."
    }
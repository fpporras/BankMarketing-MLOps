import time

import pandas as pd
import psutil


REQUEST_COUNT = 0
ERROR_COUNT = 0
LATENCIES = []


def record_request(
    latency_ms: float,
    error: bool = False
):
    """
    Registra una petición de la API.
    """

    global REQUEST_COUNT
    global ERROR_COUNT

    REQUEST_COUNT += 1

    LATENCIES.append(
        latency_ms
    )

    if error:
        ERROR_COUNT += 1


def get_system_metrics():
    """
    Obtiene métricas actuales del sistema.
    """

    cpu_percent = psutil.cpu_percent(
        interval=0.1
    )

    memory_percent = (
        psutil.virtual_memory().percent
    )

    if LATENCIES:

        average_latency = (
            sum(LATENCIES)
            / len(LATENCIES)
        )

    else:

        average_latency = 0.0

    error_rate = (

        ERROR_COUNT
        / REQUEST_COUNT

        if REQUEST_COUNT > 0

        else 0.0
    )

    return {

        "requests":
            REQUEST_COUNT,

        "errors":
            ERROR_COUNT,

        "error_rate":
            error_rate,

        "average_latency_ms":
            average_latency,

        "cpu_percent":
            cpu_percent,

        "memory_percent":
            memory_percent
    }


def monitor_system():
    """
    Ejecuta el monitoreo del sistema y devuelve
    un DataFrame para facilitar su visualización.
    """

    metrics = get_system_metrics()

    return pd.DataFrame(
        [metrics]
    )
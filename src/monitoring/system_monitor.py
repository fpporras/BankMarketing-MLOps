import time
from urllib import request, error

import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

API_URL = "http://127.0.0.1:8000/openapi.json"

NUMBER_OF_REQUESTS = 20

TIMEOUT_SECONDS = 5


# ============================================================
# SYSTEM MONITORING
# ============================================================

def monitor_system(
    api_url=API_URL,
    number_of_requests=NUMBER_OF_REQUESTS
):
    """
    Mide las métricas requeridas para O1:

    - Latency
    - Throughput
    - Error Rate
    - Availability
    """

    latencies = []
    successful_requests = 0
    failed_requests = 0

    start_test = time.perf_counter()

    for _ in range(number_of_requests):

        start_request = time.perf_counter()

        try:
            response = request.urlopen(
                api_url,
                timeout=TIMEOUT_SECONDS
            )

            status_code = response.getcode()

            if 200 <= status_code < 400:
                successful_requests += 1
            else:
                failed_requests += 1

        except (
            error.URLError,
            error.HTTPError,
            TimeoutError
        ):
            failed_requests += 1

        finally:
            end_request = time.perf_counter()

            latency = (
                end_request - start_request
            )

            latencies.append(latency)

    end_test = time.perf_counter()

    total_time = end_test - start_test

    total_requests = (
        successful_requests
        + failed_requests
    )

    # ========================================================
    # LATENCY
    # ========================================================

    average_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0
    )

    # ========================================================
    # THROUGHPUT
    # ========================================================

    throughput = (
        total_requests / total_time
        if total_time > 0
        else 0
    )

    # ========================================================
    # ERROR RATE
    # ========================================================

    error_rate = (
        failed_requests / total_requests
        if total_requests > 0
        else 0
    )

    # ========================================================
    # AVAILABILITY
    # ========================================================

    availability = (
        successful_requests / total_requests
        if total_requests > 0
        else 0
    )

    results = {
        "latency_ms": round(
            average_latency * 1000,
            2
        ),

        "throughput_req_s": round(
            throughput,
            2
        ),

        "error_rate_pct": round(
            error_rate * 100,
            2
        ),

        "availability_pct": round(
            availability * 100,
            2
        ),

        "successful_requests":
            successful_requests,

        "failed_requests":
            failed_requests,

        "total_requests":
            total_requests,
    }

    return pd.DataFrame([results])


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    report = monitor_system()

    print("\nO1 - SYSTEM MONITORING")
    print(report.to_string(index=False))
import numpy as np
import pandas as pd

def build_features(df):
    """
    Crea las variables derivadas utilizadas por el modelo.
    """

    # Copia para no modificar el DataFrame original
    df = df.copy()

    # Renombrar la columna
    df = df.rename(
        columns={
            "day_of_week": "day_of_month"
        }
    )

    # Crear indicador de contacto previo
    df["had_previous_contact"] = (
        df["pdays"] != -1
    ).astype(int)

    # Tratamiento de valores faltantes categóricos
    df["job"] = df["job"].fillna("unknown")
    df["education"] = df["education"].fillna("unknown")
    df["contact"] = df["contact"].fillna("unknown")

    df["poutcome"] = df["poutcome"].fillna(
        "no_previous_contact"
    )

    # Tratamiento de pdays
    df["pdays"] = df["pdays"].replace(-1, 0)

    # Transformaciones logarítmicas
    df["campaign_log"] = np.log1p(
        df["campaign"]
    )

    df["previous_log"] = np.log1p(
        df["previous"]
    )

    # Evitar data leakage
    if "duration" in df.columns:
        df = df.drop(
            columns=["duration"]
        )

    return df

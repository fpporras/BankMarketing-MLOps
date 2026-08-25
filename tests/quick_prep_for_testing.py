
import pandas as pd
 
RAW_PATH = "data/raw/bank_marketing.csv"
OUTPUT_PATH = "data/processed/bank_marketing_features.csv"
 
 
def main():
    # El dataset original de UCI normalmente usa ';' como separador.
    # Si tu archivo ya usa ',', cambia sep=";" por sep=",".
    df = pd.read_csv(RAW_PATH, sep=",")
    print(f"Leído: {df.shape[0]} filas, {df.shape[1]} columnas")
    print("Columnas:", list(df.columns))
 
    # Encoding simple: cada columna de texto se convierte a códigos numéricos.
    # OJO: esto es solo para probar el pipeline, NO es el Feature Engineering
    # real (no maneja 'unknown', no hace one-hot encoding, no escala nada).
    for col in df.columns:
        if df[col].dtype == object and col != "y":
            df[col] = df[col].astype("category").cat.codes
 
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Guardado en {OUTPUT_PATH}")
 
 
if __name__ == "__main__":
    main()


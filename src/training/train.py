#Importación de librerías
import pandas as pd
import numpy as np 
import logging
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
TARGET_COLUMN = "y" #Se guarda el nombre de la columna que queremos predecir 
MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "decision_tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    ),
}#Apunta a un modelo configurado con parametros inicales 
 #donde class_weight="balanced" prioriza a los clientes que dicen que si
 #y random_state=42 asegura que los resultados sean reproducibles. 

def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset ya transformado por la etapa de Feature Engineering."""
    logger.info("Cargando dataset desde %s", path)
    df = pd.read_csv(path)
    logger.info("Dataset cargado: %d filas, %d columnas", *df.shape)
    return df
 #recibe la ruta de un archivo CSV, lo lee con pandas, avisa por consola cuántas filas y columnas trae, y devuelve la tabla lista para usarse.

def split_features_target(df: pd.DataFrame):
    """Separa features (X) y variable objetivo (y)."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    if y.dtype == object:
        y = y.map({"yes": 1, "no": 0})
        if y.isna().any():
            raise ValueError(
                "La columna objetivo tiene valores distintos de 'yes'/'no' tras el mapeo."
            )

    return X, y
# Separa las tablas en dos, con X en todas las columnas menos la objetivo "y" con "y" la columna a predecir. Si "y" viene en texto lo convierte a binario y si sale algun error diferente a si o no, detiene el script. 

def evaluate(model, X_test, y_test) -> dict:
    """Calcula métricas relevantes para un problema de clasificación desbalanceado."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
#Le indico que tomé un modelo ya entrenado, le pido que prediga sobre los datos de prueba, y que calcule 5 métricas comparando lo que predijo contra lo que realmente pasó: accuracy, precision, recall, F1 y ROC-AUC  todo en un diccionario.
def make_split(df: pd.DataFrame, test_size: float = 0.2):
    """Separa X/y y hace el split estratificado train/test."""
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    logger.info(
        "Split hecho. Train: %d filas (%.1f%% positivos) | Test: %d filas (%.1f%% positivos)",
        len(X_train), y_train.mean() * 100,
        len(X_test), y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test
#Le indicó que separe X e y y luego divide todo en train (80%) y test (20%) manteniendo la misma proporción de "sí"/"no" en ambos grupos, para que el resultado sea siempre reproducible.

def train_baseline(X_train, y_train, X_test, y_test) -> dict:
    """Entrena el baseline (predice siempre la clase mayoritaria) y lo evalúa."""
    model = DummyClassifier(strategy="most_frequent", random_state=42)
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    return {
        "name": "baseline_dummy",
        "model": model,
        "params": {"strategy": "most_frequent"},
        "metrics": metrics,
    
    }
#Le indico que cree el modelo tonto (siempre predice "no"), lo entrena, lo evalúa y devuelve todo junto en un diccionario: nombre, modelo, parámetros y métricas.

def train_all_models(X_train, y_train, X_test, y_test, cv_folds: int = 5) -> list[dict]:
    """Entrena todos los modelos de MODELS con validación cruzada y los evalúa."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = []

    for name, model in MODELS.items():
        logger.info("Entrenando %s...", name)

        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1")
        logger.info("%s - F1 promedio en CV: %.4f (+/- %.4f)", name, cv_scores.mean(), cv_scores.std())

        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        metrics["cv_f1_mean"] = cv_scores.mean()
        metrics["cv_f1_std"] = cv_scores.std()

        results.append({
            "name": name,
            "model": model,
            "params": model.get_params(),
            "metrics": metrics,
        })

    return results
#Le indico que vea los 3 modelos y a cada uno 5 folds los valide antes de entrenarlo de verdad lo entrena con todo el train, lo evalúa, y guarda todo (nombre, modelo, parámetros, métricas) en una lista con un resultado por modelo.
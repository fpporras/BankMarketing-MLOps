# Bank Marketing MLOps

Proyecto de clasificación desarrollado bajo principios de MLOps para predecir si un cliente se suscribirá a un depósito bancario a partir del dataset **Bank Marketing**.

El proyecto implementa un flujo reproducible que integra ingesta de datos, diagnóstico y validación de calidad, ingeniería de variables, entrenamiento y comparación de modelos, seguimiento de experimentos con MLflow, registro y promoción del modelo, API de inferencia, contenerización con Docker, monitoreo de sistema, datos y modelo, detección de drift y una estrategia de reentrenamiento automático.

---

## 1. Business Problem

Una institución financiera realiza campañas de mercadeo directo para ofrecer depósitos bancarios a sus clientes. El problema consiste en identificar qué clientes presentan una mayor probabilidad de aceptar la suscripción.

El proyecto plantea este problema como una tarea de **clasificación binaria**.

La variable objetivo es:

- `yes`: el cliente se suscribe al depósito.
- `no`: el cliente no se suscribe.

El dataset presenta un desbalance importante entre ambas clases, debido a que la mayoría de los clientes no realiza la suscripción. Por esta razón, el desempeño del modelo no se evalúa únicamente mediante `accuracy`.

Las métricas principales utilizadas son:

- Precision
- Recall
- F1
- ROC-AUC

El objetivo final no consiste únicamente en entrenar un modelo, sino en implementar un sistema reproducible que permita entrenarlo, registrarlo, desplegarlo, monitorearlo y determinar cuándo requiere reentrenamiento.

---

## 2. Dataset

El proyecto utiliza el dataset **Bank Marketing** del **UCI Machine Learning Repository**.

La ingesta se realiza automáticamente mediante `ucimlrepo` utilizando:

```python
fetch_ucirepo(id=222)
```

El dataset raw contiene aproximadamente:

- 45,211 observaciones.
- 17 columnas.
- Variables numéricas y categóricas.
- Una variable objetivo denominada `y`.

Después de la ingesta, el archivo se almacena en:

```text
data/raw/bank_marketing.csv
```

Entre las variables utilizadas se encuentran:

```text
age
job
marital
education
default
balance
housing
loan
contact
day_of_week
month
duration
campaign
pdays
previous
poutcome
y
```

Durante el diagnóstico de calidad se revisan:

- dimensiones del dataset;
- nombres de columnas;
- tipos de datos;
- valores faltantes;
- porcentaje de valores faltantes;
- registros duplicados;
- distribución del target;
- cardinalidad;
- estadísticas numéricas;
- estadísticas categóricas;
- presencia de valores `unknown`.

También se analiza específicamente cómo se comporta la variable objetivo cuando existen valores faltantes.

---

## 3. Architecture

La arquitectura general del proyecto es la siguiente:

```text
UCI Bank Marketing
        ↓
DATA INGESTION
        ↓
data/raw/bank_marketing.csv
        ↓
DATA QUALITY DIAGNOSIS
        ↓
QUALITY GATES
        ↓
FEATURE ENGINEERING
        ↓
data/processed/df_features.csv
        ↓
PREPROCESSING
        ↓
MODEL TRAINING
        ↓
MODEL EVALUATION
        ↓
MLFLOW TRACKING
        ↓
MODEL REGISTRY
        ↓
MODEL VALIDATION
        ↓
CHAMPION
        ↓
models/best_model.joblib
        ↓
FASTAPI
        ↓
DOCKER
        ↓
PRODUCTION BATCHES
        ↓
SYSTEM MONITORING
DATA MONITORING
MODEL MONITORING
        ↓
RETRAINING DECISION
        ↓
AUTOMATIC RETRAINING
```

El proyecto utiliza un `orchestrator.py` para coordinar el pipeline principal.

El entrenamiento inicial se ejecuta mediante:

```bash
python -m src.orchestrator
```

El flujo de monitoreo y decisión de reentrenamiento se ejecuta mediante:

```bash
python -m src.orchestrator --monitor
```

El orquestador permite reutilizar el mismo pipeline tanto para el entrenamiento inicial como para un eventual reentrenamiento automático.

---

## 4. Repository Structure

La estructura principal del repositorio es:

```text
BankMarketing-MLOps/
│
├── data/
│   ├── raw/
│   │   └── bank_marketing.csv
│   │
│   ├── processed/
│   │   └── df_features.csv
│   │
│   ├── reference/
│   │   └── reference.csv
│   │
│   └── production/
│       ├── batch_1.csv
│       ├── batch_2.csv
│       └── batch_3.csv
│
├── models/
│   └── best_model.joblib
│
├── notebooks/
│   └── EDA.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   │
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── evaluation/
│   │   └── promote_model.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   └── prepare_data.py
│   │
│   ├── ingestion/
│   │   └── ingest.py
│   │
│   ├── monitoring/
│   │   ├── create_production_batch.py
│   │   ├── create_reference.py
│   │   ├── data_monitor.py
│   │   ├── model_monitor.py
│   │   ├── monitoring_runner.py
│   │   ├── simulate_drift.py
│   │   └── system_monitor.py
│   │
│   ├── preprocessing/
│   │   └── preprocess.py
│   │
│   ├── retraining/
│   │   └── retrain_decision.py
│   │
│   ├── training/
│   │   └── train.py
│   │
│   ├── validation/
│   │   ├── data_quality.py
│   │   └── quality_gates.py
│   │
│   └── orchestrator.py
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

La estructura separa las responsabilidades del sistema y evita concentrar todo el flujo de Machine Learning en un único notebook.

---

## 5. Installation

Para reproducir el proyecto desde una computadora externa se deben seguir los siguientes pasos.

### 5.1 Clonar el repositorio

Abrir una terminal y ejecutar:

```bash
git clone URL_DEL_REPOSITORIO
```

Luego ingresar al directorio:

```bash
cd BankMarketing-MLOps
```

> Sustituir `URL_DEL_REPOSITORIO` por la URL real del repositorio de GitHub.

### 5.2 Crear el entorno virtual

```bash
python -m venv .venv
```

### 5.3 Activar el entorno virtual en Windows

En PowerShell o CMD:

```bash
.venv\Scripts\activate
```

La terminal debería mostrar:

```text
(.venv)
```

### 5.4 Instalar dependencias

```bash
pip install -r requirements.txt
```

Entre las dependencias principales del proyecto se encuentran:

- pandas
- numpy
- scikit-learn
- ucimlrepo
- MLflow
- FastAPI
- Uvicorn
- joblib
- matplotlib
- psutil
- pytest

### 5.5 Flujo básico de reproducción

Una vez instalado el proyecto, el flujo principal es:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

En una segunda terminal con `.venv` activo:

```bash
python -m src.orchestrator
```

Posteriormente:

```bash
python -m src.monitoring.create_reference
```

```bash
python -m src.monitoring.create_production_batch
```

Para ejecutar la API:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Para ejecutar el monitoreo:

```bash
python -m src.orchestrator --monitor
```

---

## 6. Data Ingestion

La ingesta está implementada en:

```text
src/ingestion/ingest.py
```

El módulo descarga automáticamente el dataset Bank Marketing desde UCI mediante:

```python
fetch_ucirepo(id=222)
```

Posteriormente combina las variables predictoras y la variable objetivo en un único DataFrame.

Para ejecutar únicamente la ingesta:

```bash
python -m src.ingestion.ingest
```

El resultado se almacena en:

```text
data/raw/bank_marketing.csv
```

El módulo también informa:

- ubicación del archivo;
- número de filas;
- número de columnas;
- nombres de las variables;
- distribución del target.

El pipeline principal también ejecuta automáticamente esta etapa mediante:

```bash
python -m src.orchestrator
```

### Data Quality

Después de la ingesta se realiza un diagnóstico de calidad.

Se evalúan, entre otros aspectos:

- valores faltantes;
- duplicados;
- tipos de datos;
- cardinalidad;
- distribución de clases;
- estadísticas descriptivas;
- valores `unknown`.

### Quality Gates

Antes de permitir el entrenamiento se ejecutan ocho puertas automáticas de calidad:

1. Dataset no vacío.
2. Cantidad mínima de registros.
3. Esquema esperado.
4. Target sin valores faltantes.
5. Categorías válidas del target.
6. Edad dentro del rango permitido.
7. Tasa de duplicados.
8. Valores faltantes en variables predictoras.

Los principales umbrales configurados son:

```text
Mínimo de filas = 1000
Edad mínima = 18
Edad máxima = 100
Tasa máxima de duplicados = 1%
```

Un fallo de severidad `CRÍTICO` bloquea el pipeline.

Las advertencias permiten continuar, pero quedan registradas para tratamiento posterior.

### Feature Engineering

Una vez superadas las puertas de calidad se ejecuta:

```text
src/features/build_features.py
```

Entre las transformaciones implementadas se encuentran:

#### Indicador de contacto previo

```text
had_previous_contact
```

Se genera a partir de:

```python
pdays != -1
```

#### Tratamiento de valores faltantes

Se utilizan categorías explícitas:

```text
job       → unknown
education → unknown
contact   → unknown
poutcome  → no_previous_contact
```

#### Transformaciones logarítmicas

Se generan:

```text
campaign_log
previous_log
```

mediante `log1p`.

#### Prevención de Data Leakage

La variable:

```text
duration
```

se elimina antes del entrenamiento.

Esto se realiza porque la duración total de la llamada únicamente estaría disponible después de que la interacción ocurrió y, por tanto, utilizarla podría generar una ventaja artificial durante el entrenamiento.

El dataset procesado se almacena en:

```text
data/processed/df_features.csv
```

---

## 7. Training

El entrenamiento completo se ejecuta mediante:

```bash
python -m src.orchestrator
```

Antes de ejecutarlo se debe mantener MLflow activo en otra terminal:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

### División de datos

La configuración utilizada es:

```text
test_size = 0.20
random_state = 42
stratify = y
```

El uso de `stratify` permite conservar aproximadamente la proporción de las clases en los subconjuntos de entrenamiento y prueba.

### Preprocesamiento

Las variables numéricas son transformadas mediante:

```text
StandardScaler
```

Las variables categóricas se transforman mediante:

```text
OneHotEncoder(handle_unknown="ignore")
```

El uso de `handle_unknown="ignore"` permite que el modelo maneje categorías nuevas durante la inferencia sin producir un error.

### Baseline

Antes de entrenar los modelos candidatos se utiliza:

```text
DummyClassifier(strategy="most_frequent")
```

Este modelo funciona como línea base.

### Modelos candidatos

El proyecto compara:

```text
Logistic Regression
Decision Tree
Random Forest
```

Los modelos principales utilizan:

```text
class_weight="balanced"
```

para compensar el desbalance de la variable objetivo.

### Optimización de hiperparámetros

Se utiliza:

```text
GridSearchCV
```

con:

```text
CV_FOLDS = 5
scoring = "f1"
```

El modelo final se selecciona utilizando el mayor valor de **F1**.

### Métricas evaluadas

Para cada modelo se calculan:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
True Negatives
False Positives
False Negatives
True Positives
```

### Artefacto final

El mejor modelo queda almacenado en:

```text
models/best_model.joblib
```

Este archivo es posteriormente utilizado por la API y por el monitoreo del modelo.

---

## 8. MLflow

El proyecto utiliza **MLflow** para garantizar trazabilidad de los experimentos.

Antes del entrenamiento se debe iniciar el servidor:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Luego se puede acceder desde:

```text
http://127.0.0.1:5000
```

El tracking URI configurado en el proyecto es:

```text
http://127.0.0.1:5000
```

El experimento principal se denomina:

```text
bank-marketing-classification
```

La versión de datos registrada es:

```text
bank-marketing-v1
```

### Información registrada

MLflow registra:

- algoritmo;
- random seed;
- versión de datos;
- cantidad de features;
- feature set;
- hiperparámetros;
- métricas;
- F1 de validación cruzada;
- matriz de confusión;
- archivo de configuración;
- modelo entrenado.

### Model Registry

El mejor modelo se registra bajo el nombre:

```text
bank-marketing-classifier
```

Después del registro se realiza una validación antes de promocionarlo.

Los umbrales mínimos configurados son:

```text
F1 >= 0.40
Recall >= 0.55
ROC-AUC >= 0.70
```

Si el modelo cumple los tres criterios, la última versión registrada recibe el alias:

```text
champion
```

Esto permite distinguir formalmente el modelo aprobado para utilización.

---

## 9. Docker

Docker permite ejecutar la API dentro de un entorno aislado y reproducible.

Antes de ejecutar los siguientes comandos se debe tener **Docker Desktop** instalado y abierto.

### Construir la imagen

Desde la raíz del proyecto:

```bash
docker build -t bank-marketing-mlops .
```

El punto `.` al final indica que se utiliza la carpeta actual como contexto de construcción.

### Ejecutar el contenedor

```bash
docker run -d -p 8000:8000 --name BankMarketing bank-marketing-mlops
```

El parámetro:

```text
-p 8000:8000
```

publica el puerto del contenedor en el puerto 8000 de la computadora.

### Verificar que el contenedor esté activo

```bash
docker ps
```

Se debería observar una asignación similar a:

```text
0.0.0.0:8000->8000/tcp
```

### Abrir la API

```text
http://127.0.0.1:8000/docs
```

### Health Check

```text
http://127.0.0.1:8000/health
```

### Detener el contenedor

```bash
docker stop BankMarketing
```

### Eliminar el contenedor

```bash
docker rm BankMarketing
```

Si se desea volver a ejecutarlo:

```bash
docker run -d -p 8000:8000 --name BankMarketing bank-marketing-mlops
```

---

## 10. API

La API fue desarrollada con **FastAPI**.

El archivo principal es:

```text
src/api/main.py
```

La API carga automáticamente:

```text
models/best_model.joblib
```

al iniciar.

Para levantarla localmente sin Docker:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Luego abrir:

```text
http://127.0.0.1:8000/docs
```

### Endpoint de estado

```text
GET /health
```

Ejemplo de respuesta:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

### Endpoint de métricas

```text
GET /metrics
```

Permite consultar métricas del sistema relacionadas con:

- cantidad de requests;
- errores;
- error rate;
- latencia promedio;
- uso de CPU;
- uso de memoria.

### Endpoint de predicción

```text
POST /predict
```

Ejemplo de request:

```json
{
  "age": 35,
  "job": "technician",
  "marital": "married",
  "education": "secondary",
  "default": "no",
  "balance": 1500,
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "day_of_week": 15,
  "month": "may",
  "campaign": 2,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown"
}
```

Ejemplo de respuesta:

```json
{
  "prediction": 1,
  "probability": 0.75,
  "model_version": "1.0.0"
}
```

La probabilidad depende de los datos enviados.

### Validación de entradas

La API utiliza **Pydantic**.

Por ejemplo:

```text
age >= 18
age <= 100
```

Una entrada que no cumpla el esquema es rechazada antes de realizar la predicción.

---

## 11. Monitoring

El proyecto implementa monitoreo de:

```text
O1 - System Monitoring
O2 - Data Monitoring
O3 - Model Monitoring
O4 - Retraining Decision
```

### Preparación inicial

Antes de ejecutar el monitoreo por primera vez se debe generar un conjunto de referencia.

Ejecutar:

```bash
python -m src.monitoring.create_reference
```

Esto genera:

```text
data/reference/reference.csv
```

El conjunto de referencia corresponde a una partición estratificada utilizada como representación de la distribución base contra la cual se comparan los datos de producción.

### Crear production batches

Ejecutar:

```bash
python -m src.monitoring.create_production_batch
```

El script genera:

```text
data/production/batch_1.csv
data/production/batch_2.csv
data/production/batch_3.csv
```

Cada batch contiene una muestra de aproximadamente 1000 observaciones.

### Simulación de Drift

Para demostrar que el sistema puede detectar cambios en la distribución se puede ejecutar:

```bash
python -m src.monitoring.simulate_drift
```

La simulación modifica `batch_3.csv`.

Los cambios artificiales implementados son:

```text
age      → +15 años, limitado a 100
balance  → multiplicado por 2
campaign → +3
```

Después de esta modificación, el batch representa un escenario de producción con cambio en `P(X)`.

### O1 - System Monitoring

El sistema registra:

```text
requests
errors
error_rate
average_latency_ms
cpu_percent
memory_percent
```

La API registra el tiempo de cada request y si la petición terminó correctamente o produjo un error.

### O2 - Data Monitoring

El proyecto utiliza **Population Stability Index (PSI)**.

Las variables monitoreadas son:

```text
age
balance
campaign
pdays
previous
```

Los thresholds configurados son:

```text
PSI < 0.10            → OK

0.10 <= PSI < 0.25   → WARNING

PSI >= 0.25          → DRIFT
```

Estos umbrales se consideran reglas heurísticas de alerta y no pruebas estadísticas universales.

### O3 - Model Monitoring

Cada production batch contiene el target real `y`.

Esto permite calcular:

```text
Precision
Recall
F1
ROC-AUC
```

sobre los datos considerados de producción.

### O4 - Retraining Decision

El sistema utiliza:

```text
PSI_ALERT_THRESHOLD = 0.25
F1_MINIMUM_THRESHOLD = 0.50
```

Las reglas son:

```text
Drift + degradación de F1
→ RETRAIN = TRUE

Solo drift
→ RETRAIN = FALSE

Solo degradación del modelo
→ RETRAIN = TRUE

Sin drift ni degradación
→ RETRAIN = FALSE
```

Por lo tanto:

```text
Data Drift != Model Degradation
```

El sistema no reentrena automáticamente solo porque exista drift.

### Ejecutar Monitoring

Una vez creados `reference.csv` y los production batches:

```bash
python -m src.orchestrator --monitor
```

Este comando ejecuta:

```text
SYSTEM MONITORING
        ↓
DATA DRIFT MONITORING
        ↓
MODEL MONITORING
        ↓
RETRAINING DECISION
```

Si se cumplen las condiciones de retraining, el orquestador activa automáticamente:

```text
AUTOMATIC RETRAINING
```

y reutiliza el pipeline completo de entrenamiento.

---

## 12. Results

El proyecto permite evaluar tanto el desempeño predictivo como el comportamiento operativo del sistema.

### Resultados del modelo

En una ejecución verificada del proyecto se observaron aproximadamente:

```text
Accuracy  = 0.8437
Precision = 0.3863
Recall    = 0.5699
F1        = 0.4605
ROC-AUC   = 0.7986
```

Matriz de confusión observada:

```text
TN = 7027
FP = 958
FN = 455
TP = 603
```

La métrica principal utilizada para seleccionar el modelo es **F1**, debido al desbalance de la variable objetivo.

El ROC-AUC cercano a 0.80 muestra una capacidad razonable para discriminar entre las dos clases.

### Resultados de promoción

Para que un modelo pueda recibir el alias `champion`, debe cumplir:

```text
F1 >= 0.40
Recall >= 0.55
ROC-AUC >= 0.70
```

Cuando los tres criterios se cumplen, el modelo es promocionado en MLflow Model Registry.

### Data Drift

El sistema clasifica las variables monitoreadas en:

```text
OK
WARNING
DRIFT
```

a partir del PSI.

La simulación de producción permite comprobar si modificaciones artificiales en `age`, `balance` y `campaign` son detectadas.

### Retraining

La decisión de reentrenamiento combina:

```text
Maximum PSI
+
Current F1
```

El modelo puede reentrenarse automáticamente cuando el desempeño se encuentra por debajo del umbral configurado.

### Testing

El proyecto también cuenta con pruebas automatizadas de:

- datos;
- modelo;
- API.

Para ejecutar las pruebas:

```bash
pytest -v
```

En una ejecución verificada se obtuvo:

```text
13 passed
```

---

## 13. Team

Proyecto Integrador de Análisis de Datos / MLOps.

**Grupo 1 - Bank Marketing**

Integrantes:

- Pamela
- Fanny
- Daniela

---

### Reproducción completa del proyecto

Para una primera ejecución completa se recomienda seguir este orden.

#### Terminal 1 - MLflow

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Mantener esta terminal abierta.

#### Terminal 2 - Entrenamiento

Activar el entorno virtual:

```bash
.venv\Scripts\activate
```

Ejecutar:

```bash
python -m src.orchestrator
```

El flujo ejecutará:

```text
INGESTION
↓
DATA QUALITY
↓
QUALITY GATES
↓
FEATURE ENGINEERING
↓
TRAINING
↓
MODEL EVALUATION
↓
MLFLOW REGISTRATION
↓
MODEL VALIDATION
↓
CHAMPION
↓
best_model.joblib
```

#### Crear reference

Solo es necesario crear el dataset de referencia antes de iniciar la simulación de producción:

```bash
python -m src.monitoring.create_reference
```

Debe generarse:

```text
data/reference/reference.csv
```

#### Crear production batches

```bash
python -m src.monitoring.create_production_batch
```

Debe generar:

```text
data/production/batch_1.csv
data/production/batch_2.csv
data/production/batch_3.csv
```

#### Simular drift

```bash
python -m src.monitoring.simulate_drift
```

#### Terminal 3 - API

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Abrir:

```text
http://127.0.0.1:8000/docs
```

#### Ejecutar Monitoring y Retraining

```bash
python -m src.orchestrator --monitor
```

#### Ejecutar tests

```bash
pytest -v
```

#### Construir Docker

```bash
docker build -t bank-marketing-mlops .
```

#### Ejecutar Docker

```bash
docker run -d -p 8000:8000 --name BankMarketing bank-marketing-mlops
```

#### Verificar contenedor

```bash
docker ps
```

Abrir:

```text
http://127.0.0.1:8000/docs
```
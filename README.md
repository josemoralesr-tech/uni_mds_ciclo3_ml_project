# MLOps Final Project - Credit Default (Random Forest)

## A) Definición del problema
**Caso de uso:** Predicción de incumplimiento de pago (default) de tarjeta de crédito.  
**Objetivo:** Predecir `default.payment.next.month` (0/1) para apoyar decisiones de riesgo/seguimiento.

**Limitaciones:** clases desbalanceadas; métricas como Recall/F1 son relevantes.  
**Métricas de éxito:** ROC-AUC, F1, Recall + predicción exitosa vía API.

**Datos:**
- Raw: `data/raw/UCI_Credit_Card.csv` (si no se subió, usar dataset local)
- Dataset final: `data/training/credit_default_training.csv`

## C) Experimentación
Modelo: `RandomForestClassifier`  
Resultados: ver `reports/metrics.json`

## D) Desarrollo ML
- Preparación: `src/data_preparation.py` → `data/training/credit_default_training.csv`
- Entrenamiento: `src/train.py` → genera el modelo
- Artefactos: `models/feature_names.json`, `reports/metrics.json`

> Nota: El archivo del modelo `.joblib` no se sube al repositorio por tamaño. Para generarlo, ejecutar `python src/train.py`.

## E) Serving e inferencia
API: `src/serving.py`  
- Health: `GET /health`
- Predict: `POST /predict`  
Evidencia: `reports/inference_test.md`

## Cómo ejecutar
### 1) Crear y activar entorno
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

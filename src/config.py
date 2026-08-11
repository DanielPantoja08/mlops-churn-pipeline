"""Configuración central del proyecto.

Todas las rutas y URIs se resuelven aquí para que ningún módulo tenga rutas
hardcodeadas. Los valores se leen de variables de entorno con defaults que
funcionan en local, de modo que `python src/training/train.py` corra sin
configurar nada.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Rutas del proyecto ---------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = ROOT_DIR / "docs"
FIGURES_DIR = DOCS_DIR / "img"
MONITORING_DIR = ROOT_DIR / "monitoring"
REPORTS_DIR = MONITORING_DIR / "reports"
DRIFT_HISTORY_PATH = MONITORING_DIR / "drift_history.json"

# --- Dataset --------------------------------------------------------------

TARGET = "churn"
MONTH_COLUMN = "snapshot_month"
ID_COLUMN = "customer_id"

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "monthly_usage_gb",
    "support_tickets_30d",
    "avg_session_minutes",
    "num_services",
    "late_payments_3m",
]

CATEGORICAL_FEATURES = [
    "gender",
    "region",
    "senior_citizen",
    "contract_type",
    "payment_method",
    "paperless_billing",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Split temporal. No se usa un split aleatorio a propósito: el objetivo del
# proyecto es medir degradación en el tiempo, y mezclar meses la ocultaría.
TRAIN_MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
VALIDATION_MONTHS = ["2024-07", "2024-08"]
REFERENCE_MONTHS = TRAIN_MONTHS + VALIDATION_MONTHS  # baseline para Evidently

# --- MLflow ---------------------------------------------------------------

# Backend sqlite (no file store): el Model Registry lo exige.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "churn-prediction")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "churn-model")
CHAMPION_ALIAS = "champion"  # los stages están deprecados desde MLflow 2.9

# --- GCP emulado con Floci ------------------------------------------------

# `google-cloud-storage` y `gcsfs` leen STORAGE_EMULATOR_HOST y redirigen el
# endpoint automáticamente, además de usar credenciales anónimas. Por eso el
# mismo código sirve para Floci y para GCP real: solo cambia el entorno.
STORAGE_EMULATOR_HOST = os.getenv("STORAGE_EMULATOR_HOST", "")
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "floci-local")
DVC_BUCKET = os.getenv("DVC_BUCKET", "churn-dvc-store")
MLFLOW_ARTIFACT_BUCKET = os.getenv("MLFLOW_ARTIFACT_BUCKET", "churn-mlflow-artifacts")

# --- Monitoreo ------------------------------------------------------------

# Si más del 30 % de las columnas muestran drift significativo, se considera
# que el sistema requiere atención (y, en el pipeline completo, reentrenamiento).
DRIFT_SHARE_THRESHOLD = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.30"))

# --- API ------------------------------------------------------------------

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", f"http://localhost:{API_PORT}")


def using_emulator() -> bool:
    """True si estamos apuntando al emulador de GCP en vez de a la nube real."""
    return bool(STORAGE_EMULATOR_HOST)

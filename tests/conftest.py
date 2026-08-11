"""Fixtures compartidas por los tests.

DECISIÓN CLAVE: los tests no dependen de nada externo.

No hace falta que exista `data/raw/` (en CI los datos vienen de DVC y no se
descargan para correr tests), ni que Floci esté levantado, ni que el Model
Registry de desarrollo tenga un campeón registrado. La fixture entrena un modelo
pequeño en un registry temporal y lo registra ella misma.

Un test que necesita que alguien haya ejecutado antes un script no es un test:
es una comprobación manual con sintaxis de pytest.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Estas variables tienen que estar puestas ANTES de importar `src.config`, que
# las lee al importarse. conftest.py se carga antes que los módulos de test, así
# que este es el único sitio donde se puede hacer.
_TMP_DIR = Path(tempfile.mkdtemp(prefix="churn-tests-"))
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(_TMP_DIR / 'mlflow.db').as_posix()}"
os.environ["MLFLOW_EXPERIMENT"] = "churn-tests"
os.environ["REGISTERED_MODEL_NAME"] = "churn-model-test"
# Los tests unitarios corren sin emulador: artefactos locales y sin secretos.
os.environ.pop("STORAGE_EMULATOR_HOST", None)
os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

import mlflow  # noqa: E402
import pytest  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.config import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, TARGET  # noqa: E402
from src.data.generate_data import generate_dataset  # noqa: E402
from src.features.build_features import (  # noqa: E402
    TRUSTED_TYPES,
    build_pipeline,
    split_features_target,
)


@pytest.fixture(scope="session")
def sample_data():
    """Un dataset pequeño generado en memoria. Rápido y sin tocar disco."""
    monthly = generate_dataset(n_active=400, seed=7, n_months=4)
    import pandas as pd

    return pd.concat(monthly.values(), ignore_index=True)


@pytest.fixture(scope="session")
def registered_champion(sample_data):
    """Entrena y registra un modelo campeón en un registry temporal.

    Se usa una regresión logística sin ajustar: al test le da igual cuánto
    acierta el modelo, solo que el pipeline completo se serializa, se registra,
    se resuelve por alias y devuelve probabilidades.
    """
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT"])

    X, y = split_features_target(sample_data, TARGET)
    pipeline = build_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    pipeline.fit(X, y)

    with mlflow.start_run(run_name="fixture") as run:
        mlflow.log_param("algorithm", "logistic_regression")
        mlflow.log_metric("roc_auc", 0.85)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=X.head(2),
            code_paths=["src"],
            skops_trusted_types=TRUSTED_TYPES,
        )
        run_id = run.info.run_id

    version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    MlflowClient().set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version
    )
    return str(version.version)


@pytest.fixture(scope="session")
def client(registered_champion):
    """Cliente HTTP contra la aplicación, con el modelo ya cargado.

    `TestClient` como context manager dispara los eventos de `lifespan`, que es
    donde la API carga el modelo. Sin el `with`, el modelo nunca se cargaría y
    todos los tests darían 503.
    """
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> dict:
    from src.api.schemas import EJEMPLO_CLIENTE

    return dict(EJEMPLO_CLIENTE)

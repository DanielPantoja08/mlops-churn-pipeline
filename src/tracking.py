"""Configuración de MLflow compartida por entrenamiento, API, monitoreo y dashboard.

Dos decisiones que conviene entender:

1. **Backend sqlite, no file store.** `mlflow.set_tracking_uri("./mlruns")` es lo
   que sale en la mayoría de tutoriales, pero el Model Registry no funciona sobre
   un file store: necesita una base de datos. Y sin registry no hay alias, sin
   alias no hay promoción de modelos, y sin promoción el reentrenamiento
   automático obligaría a redesplegar la API a mano — que es justo lo que este
   proyecto quiere evitar.

2. **Alias, no stages.** Los `stages` (`Staging`, `Production`) están deprecados
   desde MLflow 2.9. El mecanismo actual son los alias: `models:/churn-model@champion`
   apunta siempre a la versión que esté marcada como campeona. Promover un modelo
   es mover ese alias — una operación atómica que la API recoge sin redespliegue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.config import (
    CHAMPION_ALIAS,
    MLFLOW_ARTIFACT_BUCKET,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    using_emulator,
)


def artifact_location() -> str | None:
    """Dónde se guardan los artefactos de los runs.

    Contra Floci (o contra GCP real) van a Cloud Storage, que es como funcionaría
    en producción. Sin emulador configurado, MLflow usa su ruta local por
    defecto — así los tests y un `git clone` limpio funcionan sin levantar nada.
    """
    if using_emulator():
        return f"gs://{MLFLOW_ARTIFACT_BUCKET}/mlflow"
    return None


def setup(experiment: str = MLFLOW_EXPERIMENT) -> str:
    """Apunta MLflow al backend correcto y asegura que el experimento existe."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    existing = client.get_experiment_by_name(experiment)
    if existing is None:
        experiment_id = client.create_experiment(experiment, artifact_location=artifact_location())
    else:
        experiment_id = existing.experiment_id

    mlflow.set_experiment(experiment)
    return experiment_id


@dataclass
class ChampionInfo:
    """Metadatos de la versión en producción, para exponerlos en `/model-info`."""

    name: str
    version: str
    run_id: str
    algorithm: str
    created_at: int
    metrics: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.name,
            "model_version": self.version,
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "created_at": self.created_at,
            "validation_metrics": self.metrics,
        }


def champion_uri(name: str = REGISTERED_MODEL_NAME) -> str:
    return f"models:/{name}@{CHAMPION_ALIAS}"


def champion_info(name: str = REGISTERED_MODEL_NAME) -> ChampionInfo:
    """Lee del registry qué versión está sirviendo y con qué métricas se eligió."""
    client = MlflowClient()
    version = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    run = client.get_run(version.run_id)
    return ChampionInfo(
        name=name,
        version=str(version.version),  # MLflow 3 lo devuelve como entero
        run_id=version.run_id,
        algorithm=run.data.params.get("algorithm", "desconocido"),
        created_at=version.creation_timestamp,
        metrics={k: round(v, 4) for k, v in run.data.metrics.items()},
    )


def load_champion(name: str = REGISTERED_MODEL_NAME):
    """Carga el modelo campeón. Se usa el flavor sklearn para tener `predict_proba`.

    `mlflow.pyfunc` solo expone `predict`, y para un problema de churn la
    probabilidad es más útil que la clase: permite ordenar clientes por riesgo y
    mover el umbral según lo que cueste una retención frente a una baja.
    """
    return mlflow.sklearn.load_model(champion_uri(name))


def promote(version: str, name: str = REGISTERED_MODEL_NAME) -> None:
    """Mueve el alias `champion` a una versión concreta.

    Esto es todo lo que hace falta para cambiar el modelo que sirve la API: la
    API resuelve el alias al arrancar y en cada recarga, sin redespliegue.
    """
    MlflowClient().set_registered_model_alias(name, CHAMPION_ALIAS, version)

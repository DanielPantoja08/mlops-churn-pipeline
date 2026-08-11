"""Prepara el emulador de GCP: crea los buckets y los secretos del proyecto.

Equivale a lo que en GCP real haría un `terraform apply` o unos cuantos
`gcloud storage buckets create`. Es idempotente: se puede ejecutar tantas veces
como haga falta.

    docker compose up -d floci-gcp
    python scripts/bootstrap_floci.py

El truco que hace que todo esto funcione sin tocar el código de la aplicación
es que tanto `google-cloud-storage` como `gcsfs` leen la variable de entorno
STORAGE_EMULATOR_HOST: si está definida, redirigen el endpoint y cambian a
credenciales anónimas por su cuenta. El mismo código sirve para Floci y para
GCP real; lo único que cambia es el entorno.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_ENDPOINT = "http://localhost:4588"

# Debe estar definida ANTES de importar los clientes de Google, porque el
# endpoint se resuelve al construir el cliente.
os.environ.setdefault("STORAGE_EMULATOR_HOST", DEFAULT_ENDPOINT)
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "floci-local")

from src.config import DVC_BUCKET, GCP_PROJECT_ID, MLFLOW_ARTIFACT_BUCKET  # noqa: E402

BUCKETS = [DVC_BUCKET, MLFLOW_ARTIFACT_BUCKET]

# Secretos que la API lee al arrancar. En GCP real vivirían en Secret Manager
# con las mismas rutas, y el código de la aplicación no cambiaría.
SECRETS = {
    "mlflow-tracking-uri": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    "churn-model-name": os.getenv("REGISTERED_MODEL_NAME", "churn-model"),
}


def ensure_buckets() -> None:
    from google.api_core import exceptions
    from google.cloud import storage

    client = storage.Client(project=GCP_PROJECT_ID)
    for name in BUCKETS:
        try:
            client.create_bucket(name)
            print(f"  [creado]  gs://{name}")
        except exceptions.Conflict:
            print(f"  [existe]  gs://{name}")


def ensure_secrets() -> None:
    from google.api_core import exceptions

    from src.gcp import secret_manager_client

    client = secret_manager_client()
    parent = f"projects/{GCP_PROJECT_ID}"

    for secret_id, value in SECRETS.items():
        try:
            client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": secret_id,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
            print(f"  [creado]  secret/{secret_id}")
        except exceptions.AlreadyExists:
            print(f"  [existe]  secret/{secret_id}")

        client.add_secret_version(
            request={
                "parent": f"{parent}/secrets/{secret_id}",
                "payload": {"data": value.encode("utf-8")},
            }
        )


def main() -> None:
    endpoint = os.environ["STORAGE_EMULATOR_HOST"]
    print(f"\nPreparando GCP emulado en {endpoint} (proyecto: {GCP_PROJECT_ID})\n")
    print("Buckets de Cloud Storage:")
    ensure_buckets()
    print("\nSecret Manager:")
    ensure_secrets()
    print("\nListo.\n")


if __name__ == "__main__":
    main()

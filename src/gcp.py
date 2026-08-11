"""Clientes de GCP que funcionan igual contra la nube real y contra el emulador.

La regla de todo el proyecto es que **el código de la aplicación no sabe si está
hablando con GCP o con Floci**. Lo único que decide eso son las variables de
entorno. Aquí está toda la lógica de esa decisión, en un solo sitio.

Los dos servicios se comportan distinto y conviene saberlo:

· Cloud Storage (REST): `google-cloud-storage` y `gcsfs` leen
  STORAGE_EMULATOR_HOST por su cuenta. Si está definida, redirigen el endpoint
  y usan credenciales anónimas sin que haya que hacer nada. Cero código.

· Secret Manager (gRPC): no lee ninguna variable de entorno y además intenta
  TLS por defecto, así que contra el emulador falla con un error de handshake
  SSL bastante críptico. Hay que construirle un canal gRPC inseguro a mano.
  De ahí que esta función exista.
"""

from __future__ import annotations

import os
from functools import lru_cache

from src.config import GCP_PROJECT_ID, STORAGE_EMULATOR_HOST


def _emulator_host() -> str:
    """Host:puerto del emulador, o cadena vacía si vamos contra GCP real.

    Se acepta SECRET_MANAGER_EMULATOR_HOST (la variable que documenta Floci)
    y, como alternativa, se deduce de STORAGE_EMULATOR_HOST, porque en Floci
    todos los servicios comparten el mismo puerto.
    """
    explicit = os.getenv("SECRET_MANAGER_EMULATOR_HOST", "")
    if explicit:
        return explicit.replace("http://", "").replace("https://", "")
    if STORAGE_EMULATOR_HOST:
        return STORAGE_EMULATOR_HOST.replace("http://", "").replace("https://", "")
    return ""


def storage_client():
    """Cliente de Cloud Storage. No necesita configuración especial."""
    from google.cloud import storage

    return storage.Client(project=GCP_PROJECT_ID)


@lru_cache(maxsize=1)
def secret_manager_client():
    """Cliente de Secret Manager, con canal inseguro si apuntamos al emulador."""
    from google.cloud import secretmanager

    host = _emulator_host()
    if not host:
        return secretmanager.SecretManagerServiceClient()

    import grpc
    from google.cloud.secretmanager_v1.services.secret_manager_service.transports import (
        SecretManagerServiceGrpcTransport,
    )

    transport = SecretManagerServiceGrpcTransport(channel=grpc.insecure_channel(host))
    return secretmanager.SecretManagerServiceClient(transport=transport)


def secret_manager_available() -> bool:
    """¿Tiene sentido siquiera intentar hablar con Secret Manager?

    Sin esta comprobación, los tests unitarios (que corren sin emulador y sin
    credenciales) construirían un cliente real, que se pone a buscar
    Application Default Credentials y a consultar el servidor de metadatos de
    GCE antes de rendirse. Son varios segundos de espera en cada test, para
    acabar en el mismo sitio: no hay secretos.
    """
    return bool(_emulator_host()) or bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))


def read_secret(secret_id: str, default: str | None = None) -> str | None:
    """Lee la última versión de un secreto, o `default` si no se puede.

    Un servicio que no arranca porque no encuentra un secreto opcional es un
    servicio frágil. Aquí la ausencia de Secret Manager degrada a la
    configuración por entorno, que es exactamente lo que se quiere en local.
    """
    if not secret_manager_available():
        return default

    name = f"projects/{GCP_PROJECT_ID}/secrets/{secret_id}/versions/latest"
    try:
        response = secret_manager_client().access_secret_version(request={"name": name})
        return response.payload.data.decode("utf-8")
    except Exception:
        return default

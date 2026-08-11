"""Fixtures de los tests de integración contra el GCP emulado.

Estos tests hablan con un servicio real por red — Floci — a diferencia de los
unitarios, que no salen del proceso. Por eso viven en su propia carpeta con su
propio conftest: los de `tests/unit` limpian a propósito las variables del
emulador para correr aislados, y heredar eso aquí rompería justo lo que se
quiere probar.

Qué aportan que los unitarios no pueden: verifican que el SDK de Google, tal y
como lo usa el código de producción, funciona contra un endpoint que no es el de
Google. Eso incluye autenticación, formatos de petición y el camino completo de
subida y descarga de artefactos de modelo. Es la parte que, sin emulador, solo
se descubre desplegando.
"""

from __future__ import annotations

import os
import socket
import time
import uuid
from urllib.parse import urlparse

import pytest

# En CI, Floci se declara como service container del job y responde en
# localhost:4588. En local es el mismo puerto vía docker compose.
DEFAULT_ENDPOINT = "http://localhost:4588"
ENDPOINT = os.getenv("GCP_EMULATOR_ENDPOINT", DEFAULT_ENDPOINT)


@pytest.fixture(scope="session", autouse=True)
def emulator_env():
    """Apunta el entorno al emulador **en tiempo de ejecución**, no al importar.

    Es importante que sea una fixture y no código a nivel de módulo. pytest
    importa TODOS los conftest durante la recolección, antes de ejecutar nada,
    así que dos conftest que tocan `os.environ` al importarse se pisan entre
    ellos y gana el último — aquí, el de `tests/unit`, que limpia a propósito
    estas mismas variables. El resultado era que la suite completa fallaba
    aunque cada carpeta pasara por separado.

    Al hacerlo en una fixture de sesión con `autouse`, cada paquete de tests
    prepara su propio entorno justo antes de usarlo.
    """
    os.environ["STORAGE_EMULATOR_HOST"] = ENDPOINT
    os.environ["SECRET_MANAGER_EMULATOR_HOST"] = urlparse(ENDPOINT).netloc
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "floci-local")
    yield


def _wait_for_emulator(endpoint: str, attempts: int = 1, delay: float = 2.0) -> bool:
    parsed = urlparse(endpoint)
    for intento in range(attempts):
        try:
            with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=2.0):
                return True
        except OSError:
            if intento < attempts - 1:
                time.sleep(delay)
    return False


@pytest.fixture(scope="session", autouse=True)
def require_emulator(emulator_env):
    """Exige el emulador, con dos comportamientos distintos según el entorno.

    · **En local**: si no está levantado, se salta la suite. Quien ejecuta
      `pytest` en su portátil sin haber hecho `docker compose up` no ha roto
      nada, y una suite en rojo por eso enseña a la gente a ignorar los rojos.

    · **En CI**: se falla. Un `skip` silencioso en CI es peor que no tener
      tests, porque el workflow sale en verde sin haber probado nada — y nadie
      revisa el recuento de tests saltados de un job que pasó. Además se espera
      con reintentos, porque un service container tarda unos segundos en
      escuchar aunque el contenedor ya esté creado.
    """
    en_ci = os.getenv("CI", "").lower() in ("1", "true")

    if _wait_for_emulator(ENDPOINT, attempts=30 if en_ci else 1):
        return

    mensaje = f"El emulador de GCP no responde en {ENDPOINT}."
    if en_ci:
        pytest.fail(
            f"{mensaje} En CI debe estar declarado como service container del job.",
            pytrace=False,
        )
    pytest.skip(f"{mensaje} Levántalo con: docker compose up -d floci-gcp")


@pytest.fixture
def bucket_name() -> str:
    """Nombre de bucket único por test, para que no interfieran entre sí."""
    return f"test-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def storage():
    from src.gcp import storage_client

    return storage_client()


@pytest.fixture
def temp_bucket(storage, bucket_name):
    bucket = storage.create_bucket(bucket_name)
    yield bucket
    try:
        for blob in list(storage.list_blobs(bucket_name)):
            blob.delete()
        bucket.delete()
    except Exception:
        pass  # el emulador es efímero; un bucket huérfano no molesta a nadie

"""Integración con Cloud Storage y Secret Manager emulados por Floci.

Lo que se valida aquí no es que Floci funcione — eso es problema de Floci — sino
que **el código de este proyecto habla correctamente con la API de GCP**: que la
subida de artefactos de modelo funciona, que los secretos se leen, y que la
degradación cuando un secreto no existe es la esperada.

Sin esto, la primera vez que se sabría si la integración con GCP funciona sería
desplegando contra la nube de verdad, gastando cuota y con credenciales reales
en un job que no las necesita.
"""

from __future__ import annotations

import contextlib
import json

import pytest

# --- Cloud Storage --------------------------------------------------------


def test_crear_bucket_y_listarlo(storage, temp_bucket):
    nombres = [b.name for b in storage.list_buckets()]
    assert temp_bucket.name in nombres


def test_subir_y_descargar_un_objeto(temp_bucket):
    contenido = json.dumps({"modelo": "churn", "version": 7})

    blob = temp_bucket.blob("modelos/metadata.json")
    blob.upload_from_string(contenido, content_type="application/json")

    descargado = temp_bucket.blob("modelos/metadata.json").download_as_text()
    assert json.loads(descargado)["version"] == 7


def test_subir_binario_del_tamano_de_un_modelo(temp_bucket):
    """Un artefacto de modelo real pesa megas, no bytes.

    Un test con una cadena de 20 caracteres no ejercita el mismo camino de
    código que una subida de verdad.
    """
    payload = b"\x00\xff" * 512 * 1024  # 1 MB

    blob = temp_bucket.blob("modelos/model.skops")
    blob.upload_from_string(payload, content_type="application/octet-stream")

    recuperado = temp_bucket.blob("modelos/model.skops").download_as_bytes()
    assert recuperado == payload
    assert len(recuperado) == 1024 * 1024


def test_listar_por_prefijo(temp_bucket):
    for i in range(3):
        temp_bucket.blob(f"runs/abc/figuras/{i}.png").upload_from_string(b"png")
    temp_bucket.blob("runs/otro/model.pkl").upload_from_string(b"pkl")

    figuras = list(temp_bucket.client.list_blobs(temp_bucket, prefix="runs/abc/figuras/"))
    assert len(figuras) == 3


def test_borrar_un_objeto(temp_bucket):
    blob = temp_bucket.blob("temporal.txt")
    blob.upload_from_string("adiós")
    assert blob.exists()

    blob.delete()
    assert not temp_bucket.blob("temporal.txt").exists()


def test_objeto_inexistente_no_existe(temp_bucket):
    assert not temp_bucket.blob("no/existe.txt").exists()


# --- Secret Manager -------------------------------------------------------


def test_leer_un_secreto_creado_por_el_bootstrap():
    """`scripts/bootstrap_floci.py` deja creado este secreto.

    Es el mismo camino de código que usa la API al arrancar para resolver la URI
    del tracking server.
    """
    from src.gcp import read_secret

    valor = read_secret("mlflow-tracking-uri")
    if valor is None:
        pytest.skip("Falta ejecutar scripts/bootstrap_floci.py contra este emulador")
    assert isinstance(valor, str)
    assert valor


def test_crear_leer_y_versionar_un_secreto():
    from google.api_core import exceptions

    from src.config import GCP_PROJECT_ID
    from src.gcp import read_secret, secret_manager_client

    client = secret_manager_client()
    parent = f"projects/{GCP_PROJECT_ID}"
    secret_id = "test-secreto-rotado"

    with contextlib.suppress(exceptions.AlreadyExists):
        client.create_secret(
            request={
                "parent": parent,
                "secret_id": secret_id,
                "secret": {"replication": {"automatic": {}}},
            }
        )

    ruta = f"{parent}/secrets/{secret_id}"
    client.add_secret_version(request={"parent": ruta, "payload": {"data": b"valor-inicial"}})
    assert read_secret(secret_id) == "valor-inicial"

    # `versions/latest` debe seguir a la versión más reciente: es lo que hace
    # que rotar un secreto no exija tocar la aplicación.
    client.add_secret_version(request={"parent": ruta, "payload": {"data": b"valor-rotado"}})
    assert read_secret(secret_id) == "valor-rotado"


def test_secreto_inexistente_devuelve_el_valor_por_defecto():
    """La API no debe caerse por un secreto opcional que falta."""
    from src.gcp import read_secret

    assert read_secret("no-existe-este-secreto", default="respaldo") == "respaldo"


# --- El camino crítico: artefactos de modelo en Cloud Storage -------------


def test_ciclo_completo_de_un_modelo_contra_cloud_storage(tmp_path, storage, bucket_name):
    """Entrena, registra en gs:// y vuelve a cargar el modelo desde el registry.

    Este es EL test de integración que importa. Recorre exactamente lo que hace
    la API al arrancar en producción: resolver un alias del Model Registry,
    descargar el artefacto de Cloud Storage y devolver probabilidades. Si algo
    va a fallar en el despliegue, falla aquí primero — sin cuota y sin
    credenciales reales.
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    from sklearn.linear_model import LogisticRegression

    from src.config import TARGET
    from src.data.generate_data import generate_dataset
    from src.features.build_features import TRUSTED_TYPES, build_pipeline, split_features_target

    storage.create_bucket(bucket_name)

    import pandas as pd

    datos = pd.concat(
        generate_dataset(n_active=300, seed=11, n_months=3).values(), ignore_index=True
    )
    X, y = split_features_target(datos, TARGET)

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    client = MlflowClient()
    experiment_id = client.create_experiment(
        "integracion-gcs", artifact_location=f"gs://{bucket_name}/mlflow"
    )

    pipeline = build_pipeline(LogisticRegression(max_iter=400, class_weight="balanced"))
    pipeline.fit(X, y)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=X.head(2),
            skops_trusted_types=TRUSTED_TYPES,
        )
        run_id = run.info.run_id

    # El artefacto tiene que estar realmente en el bucket, no en disco local.
    objetos = [b.name for b in storage.list_blobs(bucket_name)]
    assert any("model" in nombre for nombre in objetos), f"artefactos en el bucket: {objetos}"

    nombre_modelo = "churn-integracion"
    version = mlflow.register_model(f"runs:/{run_id}/model", nombre_modelo)
    client.set_registered_model_alias(nombre_modelo, "champion", version.version)

    # Y ahora el camino de la API: resolver el alias y predecir.
    recargado = mlflow.sklearn.load_model(f"models:/{nombre_modelo}@champion")
    probabilidades = recargado.predict_proba(X.head(10))[:, 1]

    assert len(probabilidades) == 10
    assert all(0.0 <= float(p) <= 1.0 for p in probabilidades)

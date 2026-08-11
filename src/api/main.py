"""API de inferencia de churn.

    uvicorn src.api.main:app --reload
    → documentación interactiva en http://localhost:8000/docs

--------------------------------------------------------------------------------
DECISIÓN CENTRAL: EL MODELO SE CARGA DEL REGISTRY, NO DE UN .pkl
--------------------------------------------------------------------------------
La API resuelve `models:/churn-model@champion` al arrancar. No hay ningún archivo
de modelo dentro de la imagen de Docker.

Parece un detalle de implementación y es lo que hace posible todo el resto del
proyecto. Con el modelo empaquetado en la imagen, promover un modelo nuevo
exigiría reconstruir la imagen y redesplegar el servicio — es decir, una persona
haciendo cosas a mano. Resolviendo un alias, promover consiste en mover ese alias
en el registry, y un `POST /reload` (o el siguiente arranque) ya sirve el modelo
nuevo. El reentrenamiento automático deja de necesitar intervención humana.

--------------------------------------------------------------------------------
DEGRADACIÓN CONTROLADA
--------------------------------------------------------------------------------
Si el modelo no se puede cargar, el proceso **no se cae**: arranca en estado
`degraded`, `/health` lo reporta con un 503 y `/predict` devuelve 503 con un
mensaje claro. Un servicio que muere en el arranque entra en un bucle de
reinicios y no le dice a nadie qué pasa; uno que arranca degradado aparece en el
panel de salud con el motivo escrito.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    Prediction,
    PredictionResponse,
)
from src.config import FEATURES, REGISTERED_MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
logger = logging.getLogger("churn-api")

# Umbral de decisión. 0.5 es el punto de partida, no una verdad: el umbral
# correcto depende de cuánto cuesta una campaña de retención frente a cuánto
# vale un cliente perdido, y esa es una decisión de negocio, no del modelo.
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

# Tramos de riesgo, para que el equipo de retención pueda priorizar sin
# interpretar probabilidades.
RISK_BANDS = ((0.35, "bajo"), (0.65, "medio"))


class ModelState:
    """Estado del modelo cargado. Se rellena en segundo plano tras el arranque."""

    def __init__(self) -> None:
        self.model: Any = None
        self.info: Any = None
        self.error: str | None = None
        self.loading: bool = False

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def version(self) -> str | None:
        return self.info.version if self.info else None

    @property
    def detail(self) -> str:
        if self.loading:
            return "Cargando el modelo desde el Model Registry..."
        return self.error or "El modelo no está cargado."


state = ModelState()


def _risk_band(probability: float) -> str:
    for limit, name in RISK_BANDS:
        if probability < limit:
            return name
    return "alto"


def load_model() -> None:
    """Carga el campeón del registry. No lanza: deja el motivo en `state.error`."""
    from src import tracking

    state.loading = True
    try:
        # La URI del tracking server puede venir de Secret Manager (emulado por
        # Floci en desarrollo y en CI, real en producción). Si no está
        # disponible, se cae al valor por defecto en vez de impedir el arranque.
        from src.gcp import read_secret

        secret_uri = read_secret("mlflow-tracking-uri")
        if secret_uri:
            os.environ["MLFLOW_TRACKING_URI"] = secret_uri
            logger.info("URI de MLflow tomada de Secret Manager")

        tracking.setup()
        state.model = tracking.load_champion()
        state.info = tracking.champion_info()
        state.error = None
        logger.info(
            "Modelo cargado: %s v%s (%s)",
            state.info.name,
            state.info.version,
            state.info.algorithm,
        )
    except Exception as exc:  # noqa: BLE001 - se registra y se sirve degradado
        state.model = None
        state.info = None
        state.error = f"{type(exc).__name__}: {exc}"
        logger.error("No se pudo cargar el modelo: %s", state.error)
    finally:
        state.loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Arranca el servidor SIN esperar a que el modelo esté cargado.

    Cargar el modelo dentro del `lifespan` de forma síncrona parecía razonable y
    resultó ser un fallo real, detectado en CI: si el Model Registry no está
    accesible, MLflow no falla al momento — reintenta con backoff exponencial
    durante varios minutos. Y mientras el `lifespan` no termina, uvicorn no
    empieza a atender peticiones, así que el servicio no responde ni siquiera a
    `/health`. El orquestador solo ve un contenedor que no contesta y lo reinicia
    en bucle, sin ninguna pista de qué está pasando.

    Es la diferencia entre un fallo rápido y uno lento: la degradación
    controlada solo funciona si el proceso llega a arrancar. Cargando en un hilo
    aparte, el servidor atiende desde el primer segundo y `/health` va contando
    la verdad — primero «cargando», y después «listo» o el error concreto.
    """
    threading.Thread(target=load_model, name="model-loader", daemon=True).start()
    yield
    state.model = None


app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan,
    description=(
        "Servicio de predicción de abandono de clientes.\n\n"
        "El modelo se resuelve desde el Model Registry de MLflow mediante el alias "
        "`@champion`, de modo que promover una versión nueva no requiere "
        "redesplegar este servicio."
    ),
)


def _require_model() -> None:
    if not state.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El modelo no está disponible. Comprueba que el Model Registry es "
                f"accesible y que existe el alias @champion. Motivo: {state.detail}"
            ),
        )


def _predict_frame(frame: pd.DataFrame) -> list[Prediction]:
    probabilities = state.model.predict_proba(frame[FEATURES])[:, 1]
    return [
        Prediction(
            churn_probability=round(float(p), 4),
            churn_prediction=int(p >= DECISION_THRESHOLD),
            risk_band=_risk_band(float(p)),
        )
        for p in probabilities
    ]


@app.get("/health", response_model=HealthResponse, tags=["operación"])
def health() -> JSONResponse:
    """Healthcheck. Devuelve 503 si el modelo no está cargado.

    El código de estado importa: es lo que miran Docker, Kubernetes o Cloud Run
    para decidir si mandar tráfico a esta instancia. Un 200 con
    `{"status": "degraded"}` en el cuerpo haría que el orquestador la considerase
    sana y le enviase peticiones que solo puede rechazar.
    """
    if state.ready:
        payload = HealthResponse(status="ok", model_loaded=True, model_version=state.version)
        return JSONResponse(status_code=200, content=payload.model_dump())

    payload = HealthResponse(status="degraded", model_loaded=False, detail=state.detail)
    return JSONResponse(status_code=503, content=payload.model_dump())


@app.get("/model-info", response_model=ModelInfoResponse, tags=["operación"])
def model_info() -> ModelInfoResponse:
    """Qué modelo está sirviendo ahora mismo y con qué métricas se eligió.

    Es el endpoint que responde a "¿esta predicción rara de qué versión salió?"
    sin tener que abrir la UI de MLflow.
    """
    _require_model()
    return ModelInfoResponse(
        **state.info.as_dict(),
        features=FEATURES,
        threshold=DECISION_THRESHOLD,
    )


@app.post("/reload", tags=["operación"])
def reload_model() -> dict[str, Any]:
    """Vuelve a resolver el alias `@champion` sin reiniciar el proceso.

    Es lo que convierte la promoción de un modelo en una operación sin
    despliegue: quien reentrena mueve el alias y llama aquí.
    """
    load_model()
    if not state.ready:
        raise HTTPException(status_code=503, detail=state.error)
    return {
        "reloaded": True,
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": state.version,
        "algorithm": state.info.algorithm,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["inferencia"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Probabilidad de abandono de un cliente."""
    _require_model()
    frame = pd.DataFrame([customer.model_dump()])
    prediction = _predict_frame(frame)[0]
    return PredictionResponse(
        **prediction.model_dump(),
        model_version=state.version,
        threshold=DECISION_THRESHOLD,
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["inferencia"])
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Igual que `/predict`, pero para varios clientes en una sola llamada.

    Una sola llamada a `predict_proba` sobre todo el lote, no una por cliente:
    la diferencia es grande porque el coste está en la sobrecarga por llamada,
    no en el cálculo.
    """
    _require_model()
    frame = pd.DataFrame([c.model_dump() for c in request.customers])
    predictions = _predict_frame(frame)
    return BatchResponse(
        predictions=predictions,
        model_version=state.version,
        threshold=DECISION_THRESHOLD,
        count=len(predictions),
    )


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": "churn-prediction-api",
        "docs": "/docs",
        "health": "/health",
    }

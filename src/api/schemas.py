"""Esquemas de entrada y salida de la API.

Los rangos que se declaran aquí no son decoración: son el contrato del servicio.
Un modelo entrenado con clientes de 18 a 92 años no tiene nada sensato que decir
sobre una edad de 400, y es mejor devolver un 422 explicando qué campo está mal
que una probabilidad inventada que alguien tomará por buena.

**Por qué unas categóricas son `Literal` y otras no.** No es una inconsistencia:

· `gender`, `senior_citizen`, `paperless_billing`, `contract_type` tienen un
  dominio cerrado y estable. Un valor fuera de esa lista es un error del cliente
  que llama, y conviene rechazarlo.
· `region` y `payment_method` cambian con el negocio: se abre una región nueva,
  se añade un método de pago. Aquí se acepta cualquier cadena razonable, porque
  el `OneHotEncoder` del pipeline lleva `handle_unknown="ignore"` y sabe
  manejarlas — las codifica como todo ceros y sigue sirviendo.

Rechazar una región nueva con un 422 obligaría a redesplegar la API cada vez que
márketing abre mercado. Aceptar una edad de 400 sería servir basura. La
diferencia está en si el valor desconocido es un error o una novedad legítima.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EJEMPLO_CLIENTE = {
    "age": 42,
    "gender": "F",
    "region": "Centro",
    "senior_citizen": "No",
    "contract_type": "Month-to-month",
    "tenure_months": 3,
    "monthly_charges": 94.5,
    "payment_method": "Electronic check",
    "paperless_billing": "Yes",
    "monthly_usage_gb": 8.2,
    "support_tickets_30d": 3,
    "avg_session_minutes": 18.4,
    "num_services": 2,
    "late_payments_3m": 2,
}


class CustomerFeatures(BaseModel):
    """Los datos de un cliente en un momento dado."""

    model_config = ConfigDict(json_schema_extra={"example": EJEMPLO_CLIENTE})

    # Demográficas
    age: int = Field(..., ge=18, le=120, description="Edad en años")
    gender: Literal["F", "M"] = Field(..., description="Género declarado")
    region: str = Field(..., min_length=1, max_length=60, description="Región comercial")
    senior_citizen: Literal["Yes", "No"] = Field(..., description="¿Tarifa de tercera edad?")

    # Contractuales
    contract_type: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Tipo de contrato. Es el predictor más fuerte del modelo"
    )
    tenure_months: int = Field(..., ge=0, le=600, description="Meses de antigüedad")
    monthly_charges: float = Field(..., gt=0, le=1000, description="Cargo mensual")
    payment_method: str = Field(..., min_length=1, max_length=60, description="Método de pago")
    paperless_billing: Literal["Yes", "No"] = Field(..., description="¿Factura electrónica?")

    # Comportamiento
    monthly_usage_gb: float = Field(..., ge=0, le=10_000, description="Consumo mensual en GB")
    support_tickets_30d: int = Field(
        ..., ge=0, le=100, description="Tickets de soporte en los últimos 30 días"
    )
    avg_session_minutes: float = Field(
        ..., ge=0, le=1440, description="Duración media de sesión en minutos"
    )
    num_services: int = Field(..., ge=1, le=20, description="Número de servicios contratados")
    late_payments_3m: int = Field(..., ge=0, le=50, description="Impagos en los últimos 3 meses")


class BatchRequest(BaseModel):
    """Varios clientes de una vez.

    El límite de 1.000 evita que una petición mal formada bloquee el worker.
    Para volúmenes mayores, lo correcto es un job por lotes, no una llamada HTTP.
    """

    model_config = ConfigDict(json_schema_extra={"example": {"customers": [EJEMPLO_CLIENTE]}})

    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    churn_probability: float = Field(..., description="Probabilidad estimada de abandono")
    churn_prediction: int = Field(..., description="Clase al umbral aplicado (0 o 1)")
    risk_band: Literal["bajo", "medio", "alto"] = Field(
        ...,
        description=(
            "Tramo de riesgo. Se devuelve además de la probabilidad porque quien "
            "consume esto suele ser un equipo de retención que necesita priorizar, "
            "no interpretar un número entre 0 y 1"
        ),
    )


class PredictionResponse(Prediction):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "churn_probability": 0.7412,
                "churn_prediction": 1,
                "risk_band": "alto",
                "model_version": "1",
                "threshold": 0.5,
            }
        }
    )

    model_version: str = Field(..., description="Versión del modelo que atendió la petición")
    threshold: float = Field(..., description="Umbral usado para convertir probabilidad en clase")


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
    threshold: float
    count: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"status": "ok", "model_loaded": True, "model_version": "1"}}
    )

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    run_id: str
    algorithm: str
    created_at: int
    validation_metrics: dict[str, float]
    features: list[str]
    threshold: float

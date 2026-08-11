"""Tests de la API de inferencia."""

from __future__ import annotations

import pytest


# --- Operación -----------------------------------------------------------


def test_health_ok_con_modelo_cargado(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_model_info_expone_version_y_metricas(client, registered_champion):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()

    assert body["model_version"] == registered_champion
    assert body["algorithm"] == "logistic_regression"
    assert "roc_auc" in body["validation_metrics"]
    # Las features declaradas son las que el modelo espera: si alguien añade una
    # al config y no reentrena, este test lo detecta antes que producción.
    assert len(body["features"]) == 14
    assert "customer_id" not in body["features"]


def test_reload_vuelve_a_resolver_el_alias(client):
    response = client.post("/reload")
    assert response.status_code == 200
    assert response.json()["reloaded"] is True


# --- Predicción individual ------------------------------------------------


def test_predict_con_entrada_valida(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()

    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["risk_band"] in ("bajo", "medio", "alto")
    assert body["model_version"]


def test_prediccion_coherente_con_umbral(client, valid_payload):
    """La clase devuelta tiene que ser consistente con la probabilidad.

    Parece trivial y es justo el tipo de desajuste que se cuela al tocar el
    umbral: una respuesta que dice probabilidad 0.8 y clase 0.
    """
    body = client.post("/predict", json=valid_payload).json()
    esperado = int(body["churn_probability"] >= body["threshold"])
    assert body["churn_prediction"] == esperado


def test_perfiles_de_riesgo_opuestos_se_ordenan_bien(client, valid_payload):
    """Un cliente claramente de alto riesgo debe puntuar por encima de uno leal.

    Este test comprueba que el pipeline serializado aplica de verdad las
    transformaciones. Si el escalado se perdiera al serializar, las predicciones
    seguirían siendo números entre 0 y 1 — no saltaría ninguna excepción — pero
    dejarían de tener sentido. Es exactamente el fallo silencioso que el
    training/serving skew produce.
    """
    riesgo_alto = valid_payload | {
        "contract_type": "Month-to-month",
        "tenure_months": 1,
        "monthly_charges": 120.0,
        "support_tickets_30d": 6,
        "late_payments_3m": 3,
    }
    riesgo_bajo = valid_payload | {
        "contract_type": "Two year",
        "tenure_months": 60,
        "monthly_charges": 35.0,
        "support_tickets_30d": 0,
        "late_payments_3m": 0,
    }

    p_alto = client.post("/predict", json=riesgo_alto).json()["churn_probability"]
    p_bajo = client.post("/predict", json=riesgo_bajo).json()["churn_probability"]

    assert p_alto > p_bajo


# --- Validación de entrada -------------------------------------------------


def test_campo_faltante_devuelve_422(client, valid_payload):
    del valid_payload["contract_type"]
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422
    assert any(e["loc"][-1] == "contract_type" for e in response.json()["detail"])


def test_tipo_invalido_devuelve_422(client, valid_payload):
    valid_payload["age"] = "cuarenta y dos"
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_valor_fuera_de_rango_devuelve_422(client, valid_payload):
    """El modelo se entrenó con edades de 18 a 92: 400 no es un cliente."""
    valid_payload["age"] = 400
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_categoria_cerrada_invalida_devuelve_422(client, valid_payload):
    valid_payload["contract_type"] = "Contrato vitalicio"
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_categoria_abierta_desconocida_se_acepta(client, valid_payload):
    """Una región nueva no debe tumbar el servicio.

    Es la contrapartida del test anterior y una decisión de diseño explícita: el
    `OneHotEncoder` lleva `handle_unknown="ignore"`, así que márketing puede
    abrir una región sin que haya que redesplegar la API.
    """
    valid_payload["region"] = "Antártida"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_probability"] <= 1.0


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("monthly_charges", -10.0),
        ("tenure_months", -1),
        ("support_tickets_30d", -3),
        ("num_services", 0),
    ],
)
def test_valores_negativos_o_imposibles_devuelven_422(client, valid_payload, campo, valor):
    valid_payload[campo] = valor
    assert client.post("/predict", json=valid_payload).status_code == 422


# --- Predicción por lotes --------------------------------------------------


def test_batch_devuelve_una_prediccion_por_cliente(client, valid_payload):
    response = client.post("/predict/batch", json={"customers": [valid_payload] * 5})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 5
    assert len(body["predictions"]) == 5


def test_batch_coincide_con_predict_individual(client, valid_payload):
    """Predecir en lote y de uno en uno debe dar exactamente lo mismo."""
    individual = client.post("/predict", json=valid_payload).json()["churn_probability"]
    lote = client.post("/predict/batch", json={"customers": [valid_payload]}).json()
    assert lote["predictions"][0]["churn_probability"] == individual


def test_batch_vacio_devuelve_422(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


# --- Degradación controlada ------------------------------------------------


def test_predict_devuelve_503_sin_modelo(client, valid_payload, monkeypatch):
    """Sin modelo cargado, el servicio responde 503, no 500 ni una predicción falsa."""
    from src.api import main

    monkeypatch.setattr(main.state, "model", None)
    monkeypatch.setattr(main.state, "error", "simulado para el test")

    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 503
    assert "simulado para el test" in response.json()["detail"]

    salud = client.get("/health")
    assert salud.status_code == 503
    assert salud.json()["status"] == "degraded"

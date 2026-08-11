"""Tests del dashboard de Streamlit.

Que `streamlit run` arranque y el healthcheck devuelva 200 NO significa que el
dashboard funcione: Streamlit ejecuta el script cuando se conecta una sesión, así
que una excepción en el código de la página no aparece hasta que alguien abre el
navegador. Un dashboard roto que responde 200 al healthcheck es exactamente el
tipo de fallo que se descubre durante la demo.

`AppTest` ejecuta el script de verdad, sin navegador, y expone lo que se ha
renderizado.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit", reason="el grupo 'dashboard' no está instalado")

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.config import DRIFT_HISTORY_PATH, ROOT_DIR  # noqa: E402

APP = ROOT_DIR / "dashboard" / "app.py"

pytestmark = pytest.mark.skipif(
    not DRIFT_HISTORY_PATH.exists(),
    reason="hace falta monitoring/drift_history.json (python monitoring/generate_report.py)",
)


@pytest.fixture(scope="module")
def app():
    """Ejecuta el dashboard completo.

    No hay API levantada, y es intencionado: el dashboard tiene que renderizar
    igualmente y decir que la API no responde, en vez de reventar. Es el mismo
    principio de degradación controlada que aplica la API con el modelo.
    """
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    return at


def test_el_dashboard_se_renderiza_sin_excepciones(app):
    assert not app.exception, f"el dashboard lanzó: {app.exception}"


def test_muestra_el_titulo(app):
    assert any("churn" in t.value.lower() for t in app.title)


def test_muestra_las_cuatro_pestanas(app):
    assert len(app.tabs) >= 4


def test_expone_el_estado_del_modelo_en_metricas(app):
    etiquetas = [m.label for m in app.metric]
    assert any("AUC" in e for e in etiquetas)
    assert any("Modelo" in e for e in etiquetas)


def test_avisa_de_la_degradacion_del_ultimo_periodo(app):
    """El último mes del histórico está en concept drift: debe salir en rojo.

    Comprueba que el semáforo refleja los datos reales y no un valor fijo.
    """
    import json

    historico = json.loads(DRIFT_HISTORY_PATH.read_text(encoding="utf-8"))
    estado = historico["history"][-1]["status"]

    if estado == "CRITICO":
        assert app.error, "un estado CRITICO debería mostrar una alerta de error"
        assert any("degradado" in e.value.lower() for e in app.error)
    elif estado == "VIGILAR":
        assert app.warning


def test_informa_del_estado_de_la_api(app):
    """El panel reporta el estado real de la API, esté levantada o no.

    El test no asume ninguno de los dos casos a propósito: se ejecuta igual en
    CI (sin API) que en local con el stack de Docker en marcha. Lo que verifica
    es que el estado se muestra siempre y que es uno de los contemplados — un
    dashboard que se cae porque la API no responde no sirve justo cuando más
    falta hace.
    """
    valores = [str(m.value) for m in app.metric]
    estados = {"en línea", "degradada", "sin respuesta"}
    assert estados & set(valores), f"ningún estado de API reconocible en {valores}"

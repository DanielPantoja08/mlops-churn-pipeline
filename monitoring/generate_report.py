"""Monitoreo de drift con Evidently AI.

    python monitoring/generate_report.py

Compara cada mes posterior al entrenamiento contra el periodo de referencia y
produce, por mes, un informe HTML navegable, un JSON consultable y una entrada en
`monitoring/drift_history.json`.

--------------------------------------------------------------------------------
POR QUÉ SE MIDEN TRES SEÑALES Y NO UNA
--------------------------------------------------------------------------------
Un sistema de monitoreo que vigila una sola cosa se equivoca en una de las dos
direcciones: o no detecta degradaciones, o dispara alarmas cuando no pasa nada.
Aquí se recogen tres señales que responden a preguntas distintas:

1. **Drift de datos** — ¿ha cambiado la distribución de las entradas?
   Disponible siempre, en tiempo real, sin necesidad de etiquetas.
   Pero un cambio de distribución NO implica que el modelo esté peor.

2. **Drift de predicciones** — ¿ha cambiado la distribución de lo que el modelo
   predice? También disponible sin etiquetas. Es más informativa que la anterior
   porque solo se mueve si el cambio de entrada afecta a la salida: si el drift
   ocurre en una variable a la que el modelo apenas presta atención, esta señal
   ni se inmuta.

3. **Rendimiento real (AUC)** — la verdad, pero llega tarde.
   *Advertencia importante y deliberada:* aquí se puede calcular al instante
   porque el dataset es sintético y las etiquetas existen desde el principio. En
   producción no es así — saber si un cliente se ha ido tarda semanas o meses.
   Ese retraso es justamente la razón de que existan las dos primeras señales.
   Presentar el AUC mensual como si estuviera disponible en vivo sería
   deshonesto; se incluye como referencia de VALIDACIÓN, para poder comprobar si
   las señales que sí están disponibles a tiempo habrían detectado el problema.

--------------------------------------------------------------------------------
CÓMO SE LEEN JUNTAS
--------------------------------------------------------------------------------
    Drift de datos   Rendimiento   Interpretación
    ---------------  ------------  --------------------------------------------
    No               Estable       Todo en orden
    SÍ               Estable       Cambió la población, el modelo sigue siendo
                                   válido → vigilar, NO reentrenar
    Sí o no          CAE           Concept drift → reentrenar

El caso interesante es el segundo. Reentrenar con cada alerta de drift de datos
es quemar dinero y arriesgar un modelo que funcionaba.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from evidently import Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import tracking  # noqa: E402
from src.config import DRIFT_HISTORY_PATH as HISTORY_PATH  # noqa: E402
from src.config import (  # noqa: E402
    DRIFT_SHARE_THRESHOLD,
    FEATURES,
    REPORTS_DIR,
    TARGET,
    TRAIN_MONTHS,
)
from src.data.generate_data import load_raw, regime_for  # noqa: E402
from src.viz import (  # noqa: E402
    INK_MUTED,
    REGIME_COLORS,
    SERIES,
    STATUS,
    save_figure,
    set_style,
)

# Caída de AUC respecto a la referencia a partir de la cual se considera que el
# modelo se ha degradado de verdad y no es ruido de muestreo mensual.
AUC_DROP_THRESHOLD = 0.02

PREDICTION_COLUMN = "prediction"


@dataclass
class MonthlyDrift:
    """Resultado del monitoreo de un mes."""

    month: str
    regime: str
    n_rows: int
    drifted_columns_count: int
    drifted_columns_share: float
    drifted_columns: list[str]
    prediction_drift_score: float | None
    prediction_drift_detected: bool
    roc_auc: float | None
    auc_delta: float | None
    status: str
    report_html: str
    report_json: str


def _extract_column_drift(snapshot_dict: dict) -> tuple[dict[str, bool], dict[str, float]]:
    """Saca de la salida de Evidently qué columnas han derivado.

    Se decide por el resultado del TEST asociado a cada métrica, no comparando
    el valor con el umbral a mano. El motivo: Evidently elige el método según el
    tipo de columna (Wasserstein para numéricas, Jensen-Shannon para
    categóricas, y tests estadísticos en otros casos), y no todos los métodos
    apuntan en la misma dirección — en una distancia, más alto significa más
    drift; en un p-valor, es al revés. Delegar en el test evita esa clase de
    error silencioso.
    """
    por_id: dict[str, tuple[str, float]] = {}
    for metric in snapshot_dict.get("metrics", []):
        config = metric.get("config", {})
        if not config.get("type", "").endswith("ValueDrift"):
            continue
        column = config.get("column")
        value = metric.get("value")
        if column is not None and isinstance(value, (int, float)):
            por_id[metric["id"]] = (column, float(value))

    estado_test: dict[str, bool] = {}
    for test in snapshot_dict.get("tests", []):
        metric_id = test.get("metric_config", {}).get("metric_id")
        if metric_id in por_id:
            # SUCCESS = el test pasa = NO hay drift.
            estado_test[metric_id] = test.get("status") != "SUCCESS"

    derivadas: dict[str, bool] = {}
    puntuaciones: dict[str, float] = {}
    for metric_id, (column, value) in por_id.items():
        puntuaciones[column] = round(value, 4)
        if metric_id in estado_test:
            derivadas[column] = estado_test[metric_id]
        else:
            # Respaldo si el preset no generase test para esa columna.
            umbral = 0.1
            derivadas[column] = value > umbral

    return derivadas, puntuaciones


def _status(drift_share: float, auc_delta: float | None) -> str:
    """Traduce las señales a un estado accionable.

    El orden de las comprobaciones es la política: una caída de rendimiento manda
    sobre cualquier otra cosa, y el drift de datos por sí solo nunca llega a
    CRÍTICO — solo a VIGILAR. Es la traducción a código de la tabla del docstring.
    """
    if auc_delta is not None and auc_delta <= -AUC_DROP_THRESHOLD:
        return "CRITICO"
    if drift_share > DRIFT_SHARE_THRESHOLD:
        return "VIGILAR"
    return "OK"


def analyse_month(
    month: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_auc: float | None,
    month_index: int,
) -> MonthlyDrift:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Se comparan las features Y la predicción del modelo. Incluir la predicción
    # es lo que da la señal de prediction drift, que está disponible sin
    # etiquetas y por tanto sirve en producción desde el primer día.
    columnas = [*FEATURES, PREDICTION_COLUMN]
    snapshot = Report([DataDriftPreset()], include_tests=True).run(
        current_data=current[columnas],
        reference_data=reference[columnas],
    )

    ruta_html = REPORTS_DIR / f"drift_{month}.html"
    ruta_json = REPORTS_DIR / f"drift_{month}.json"
    snapshot.save_html(str(ruta_html))

    datos = snapshot.dict()
    ruta_json.write_text(json.dumps(datos, indent=2, default=str), encoding="utf-8")

    derivadas, puntuaciones = _extract_column_drift(datos)

    # El recuento se hace solo sobre las FEATURES: la columna de predicción se
    # informa aparte, porque mezclarla inflaría artificialmente el porcentaje.
    derivadas_features = [c for c in FEATURES if derivadas.get(c)]
    share = len(derivadas_features) / len(FEATURES)

    auc = None
    delta = None
    if TARGET in current.columns and current[TARGET].nunique() > 1:
        auc = float(roc_auc_score(current[TARGET], current[PREDICTION_COLUMN]))
        if reference_auc is not None:
            delta = auc - reference_auc

    return MonthlyDrift(
        month=month,
        regime=regime_for(month_index),
        n_rows=len(current),
        drifted_columns_count=len(derivadas_features),
        drifted_columns_share=round(share, 4),
        drifted_columns=derivadas_features,
        prediction_drift_score=puntuaciones.get(PREDICTION_COLUMN),
        prediction_drift_detected=bool(derivadas.get(PREDICTION_COLUMN, False)),
        roc_auc=round(auc, 4) if auc is not None else None,
        auc_delta=round(delta, 4) if delta is not None else None,
        status=_status(share, delta),
        report_html=str(ruta_html.relative_to(HISTORY_PATH.parent.parent)).replace("\\", "/"),
        report_json=str(ruta_json.relative_to(HISTORY_PATH.parent.parent)).replace("\\", "/"),
    )


def _history_figure(history: list[MonthlyDrift]):
    """Las dos señales en el tiempo, una encima de otra y con el eje X compartido.

    Deliberadamente NO es un gráfico de doble eje Y. Superponer dos escalas
    distintas en un mismo eje invita a leer cruces entre curvas que no
    significan nada. Aquí lo que hay que comparar es *cuándo* se mueve cada
    señal, y para eso basta con compartir el eje temporal.
    """
    meses = [h.month for h in history]
    x = np.arange(len(meses))

    fig, (ax_drift, ax_auc) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"hspace": 0.18}
    )

    # --- Arriba: proporción de columnas con drift
    colores = [
        STATUS["critical"]
        if h.status == "CRITICO"
        else (STATUS["warning"] if h.status == "VIGILAR" else STATUS["good"])
        for h in history
    ]
    ax_drift.bar(
        x, [h.drifted_columns_share for h in history], color=colores, width=0.68, linewidth=0
    )
    ax_drift.axhline(DRIFT_SHARE_THRESHOLD, color=INK_MUTED, linestyle="--", linewidth=1.2)
    ax_drift.annotate(
        f"umbral {DRIFT_SHARE_THRESHOLD:.0%}",
        xy=(len(meses) - 0.4, DRIFT_SHARE_THRESHOLD),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=9,
        color=INK_MUTED,
    )
    ax_drift.set_ylim(0, 0.55)
    ax_drift.set_ylabel("Columnas con drift")
    ax_drift.set_title("Señal 1 · drift de datos (disponible sin etiquetas, en tiempo real)")

    # Los regímenes se etiquetan solo en el panel superior: repetirlos abajo
    # sería ruido, y el eje X es compartido.
    for regimen, etiqueta in (
        ("baseline", "Baseline"),
        ("data_drift", "Data drift"),
        ("concept_drift", "Concept drift"),
    ):
        posiciones = [i for i, h in enumerate(history) if h.regime == regimen]
        if posiciones:
            ax_drift.annotate(
                etiqueta,
                xy=(float(np.mean(posiciones)), 0.53),
                ha="center",
                va="top",
                fontsize=9,
                fontweight="semibold",
                color=REGIME_COLORS[regimen],
            )

    # --- Abajo: AUC real
    aucs = [h.roc_auc for h in history]
    ax_auc.plot(x, aucs, marker="o", color=SERIES[0], zorder=3)
    referencia = aucs[0]
    if referencia is not None:
        ax_auc.axhline(referencia, color=INK_MUTED, linestyle="--", linewidth=1.2)
        ax_auc.axhspan(
            referencia - AUC_DROP_THRESHOLD,
            referencia + AUC_DROP_THRESHOLD,
            color=STATUS["good"],
            alpha=0.10,
            linewidth=0,
        )
    ax_auc.set_ylabel("AUC-ROC")
    ax_auc.set_ylim(0.72, 0.92)
    ax_auc.set_title("Señal 2 · rendimiento real (en producción llegaría con semanas de retraso)")
    ax_auc.set_xticks(x, meses, rotation=45, ha="right")

    indices = [m for m, _ in enumerate(meses)]
    offset = 18 - len(meses)  # el histórico empieza después de los meses de train
    etiquetas_regimen = [regime_for(i + offset) for i in indices]
    _shade(ax_drift, etiquetas_regimen)
    _shade(ax_auc, etiquetas_regimen)

    fig.suptitle(
        "El drift de datos se dispara antes que la degradación — y no siempre la anuncia",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _shade(ax, regimenes: list[str]) -> None:
    inicio = 0
    actual = regimenes[0]
    for i in range(1, len(regimenes) + 1):
        if i == len(regimenes) or regimenes[i] != actual:
            ax.axvspan(
                inicio - 0.5,
                i - 0.5,
                color=REGIME_COLORS[actual],
                alpha=0.07,
                zorder=0,
                linewidth=0,
            )
            if i < len(regimenes):
                inicio, actual = i, regimenes[i]


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera los informes de drift")
    parser.add_argument(
        "--months",
        nargs="*",
        default=None,
        help="meses concretos a analizar; por defecto, todos los posteriores al entrenamiento",
    )
    args = parser.parse_args()

    set_style()
    tracking.setup()

    print("Cargando el modelo campeón del registry...")
    model = tracking.load_champion()
    info = tracking.champion_info()
    print(f"  {info.name} v{info.version} ({info.algorithm})\n")

    datos = load_raw()
    todos_los_meses = sorted(datos.snapshot_month.unique())

    # El modelo puntúa TODAS las filas de una vez: así la columna de predicción
    # es comparable entre la referencia y cada mes.
    datos[PREDICTION_COLUMN] = model.predict_proba(datos[FEATURES])[:, 1]

    referencia = datos[datos.snapshot_month.isin(TRAIN_MONTHS)]
    auc_referencia = float(roc_auc_score(referencia[TARGET], referencia[PREDICTION_COLUMN]))
    print(f"Referencia: {TRAIN_MONTHS[0]}..{TRAIN_MONTHS[-1]}  ·  AUC {auc_referencia:.4f}\n")

    a_analizar = args.months or [m for m in todos_los_meses if m not in TRAIN_MONTHS]

    historico: list[MonthlyDrift] = []
    print(f"{'mes':>9} {'regimen':>14} {'drift':>7} {'pred':>6} {'AUC':>7} {'delta':>8}  estado")
    print("-" * 72)

    for mes in a_analizar:
        actual = datos[datos.snapshot_month == mes]
        resultado = analyse_month(
            month=mes,
            reference=referencia,
            current=actual,
            reference_auc=auc_referencia,
            month_index=todos_los_meses.index(mes),
        )
        historico.append(resultado)
        print(
            f"{resultado.month:>9} {resultado.regime:>14} "
            f"{resultado.drifted_columns_share:>6.0%} "
            f"{'sí' if resultado.prediction_drift_detected else 'no':>6} "
            f"{resultado.roc_auc:>7.4f} {resultado.auc_delta:>+8.4f}  {resultado.status}"
        )

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(
        json.dumps(
            {
                "model": {
                    "name": info.name,
                    "version": info.version,
                    "algorithm": info.algorithm,
                },
                "reference_months": TRAIN_MONTHS,
                "reference_roc_auc": round(auc_referencia, 4),
                "thresholds": {
                    "drift_share": DRIFT_SHARE_THRESHOLD,
                    "auc_drop": AUC_DROP_THRESHOLD,
                },
                "history": [asdict(h) for h in historico],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    figura = _history_figure(historico)
    save_figure(figura, "08_historico_drift")
    plt.close(figura)

    print(f"\nInformes HTML y JSON en {REPORTS_DIR}")
    print(f"Histórico consolidado en {HISTORY_PATH}")

    primer_vigilar = next((h.month for h in historico if h.status == "VIGILAR"), None)
    primer_critico = next((h.month for h in historico if h.status == "CRITICO"), None)
    print(
        "\nLectura:"
        f"\n  · Primera alerta de drift de datos : {primer_vigilar}"
        f"\n  · Primera degradación real         : {primer_critico}"
        "\n  · La distancia entre ambas fechas es el argumento del proyecto: hubo"
        "\n    drift de datos durante meses sin que el modelo empeorase.\n"
    )


if __name__ == "__main__":
    main()

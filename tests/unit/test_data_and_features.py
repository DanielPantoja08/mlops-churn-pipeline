"""Tests del generador de datos y del pipeline de features.

Los tests sobre el generador no son de relleno: **convierten en ejecutables las
afirmaciones centrales del proyecto**. El README dice que hay data drift en
2024-09 y concept drift en 2025-01; aquí eso deja de ser una frase en un
documento y pasa a ser algo que CI comprueba en cada push.

Si alguien toca los coeficientes y rompe la narrativa, se entera aquí y no
delante de un entrevistador.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET
from src.data.generate_data import (
    CONCEPT_DRIFT_START,
    DATA_DRIFT_START,
    generate_dataset,
    month_labels,
    regime_for,
)
from src.features.build_features import Winsorizer, build_pipeline, split_features_target


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """18 meses con una base reducida: suficiente para que la señal se vea."""
    monthly = generate_dataset(n_active=900, seed=42, n_months=18)
    frames = []
    for idx, (label, frame) in enumerate(monthly.items()):
        frame = frame.copy()
        frame["month_index"] = idx
        frame["regimen"] = regime_for(idx)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --- Estructura y reproducibilidad ----------------------------------------


def test_el_generador_es_reproducible():
    """Misma semilla, mismos datos. Sin esto no hay experimento replicable."""
    a = generate_dataset(n_active=200, seed=123, n_months=3)
    b = generate_dataset(n_active=200, seed=123, n_months=3)
    for label in a:
        pd.testing.assert_frame_equal(a[label], b[label])


def test_semillas_distintas_dan_datos_distintos():
    a = generate_dataset(n_active=200, seed=1, n_months=2)
    b = generate_dataset(n_active=200, seed=2, n_months=2)
    assert not a["2024-01"].equals(b["2024-01"])


def test_las_etiquetas_de_mes_son_consecutivas():
    labels = month_labels(n_months=18)
    assert labels[0] == "2024-01"
    assert labels[8] == "2024-09"  # inicio del data drift
    assert labels[12] == "2025-01"  # inicio del concept drift
    assert labels[-1] == "2025-06"


def test_los_regimenes_empiezan_donde_dice_la_documentacion():
    assert regime_for(0) == "baseline"
    assert regime_for(DATA_DRIFT_START - 1) == "baseline"
    assert regime_for(DATA_DRIFT_START) == "data_drift"
    assert regime_for(CONCEPT_DRIFT_START - 1) == "data_drift"
    assert regime_for(CONCEPT_DRIFT_START) == "concept_drift"


def test_no_hay_nulos_ni_duplicados_de_clave(dataset):
    assert dataset[FEATURES + [TARGET]].isna().sum().sum() == 0
    assert not dataset.duplicated(subset=["customer_id", "snapshot_month"]).any()


def test_el_dataset_es_un_panel(dataset):
    """Un cliente que no abandona reaparece el mes siguiente."""
    apariciones = dataset.groupby("customer_id").size()
    assert (apariciones > 1).sum() > 0


# --- Las afirmaciones centrales del proyecto ------------------------------


def test_la_tasa_base_de_churn_es_estable(dataset):
    """La calibración por mes tiene que mantener la prevalencia fija.

    Es el control del experimento: si la tasa se moviera, no se podría atribuir
    una caída de rendimiento a un cambio de relación en vez de a un cambio de
    prevalencia.
    """
    por_mes = dataset.groupby("month_index")[TARGET].mean()
    assert por_mes.min() > 0.15
    assert por_mes.max() < 0.26
    assert por_mes.std() < 0.02


def test_hay_data_drift_en_el_uso_mensual(dataset):
    """`monthly_usage_gb` sube claramente a partir de 2024-09."""
    baseline = dataset.loc[dataset.regimen == "baseline", "monthly_usage_gb"].mean()
    posterior = dataset.loc[dataset.regimen != "baseline", "monthly_usage_gb"].mean()
    assert posterior > baseline * 1.4


def test_el_data_drift_es_una_rampa_no_un_escalon(dataset):
    """Un escalón haría trivial la detección; el drift real es progresivo."""
    medias = dataset.groupby("month_index").monthly_usage_gb.mean()
    rampa = medias.loc[DATA_DRIFT_START:CONCEPT_DRIFT_START - 1]
    assert rampa.is_monotonic_increasing
    # Ningún mes concentra el salto entero.
    salto_total = medias.iloc[-1] - medias.iloc[0]
    assert rampa.diff().max() < salto_total * 0.6


def test_el_metodo_de_pago_tambien_deriva(dataset):
    cuota_baseline = (
        dataset.loc[dataset.regimen == "baseline", "payment_method"] == "Electronic check"
    ).mean()
    cuota_final = (
        dataset.loc[dataset.regimen == "concept_drift", "payment_method"] == "Electronic check"
    ).mean()
    assert cuota_final < cuota_baseline * 0.8


def test_hay_concept_drift_en_los_tickets_de_soporte(dataset):
    """LA afirmación central: la relación tickets-churn se invierte en 2025-01.

    Durante el baseline y todo el data drift, la correlación es claramente
    positiva. Con el concept drift deja de serlo.
    """
    correlaciones = dataset.groupby("regimen").apply(
        lambda g: g["support_tickets_30d"].corr(g[TARGET]), include_groups=False
    )
    assert correlaciones["baseline"] > 0.15
    assert correlaciones["data_drift"] > 0.15  # el data drift NO rompe la relación
    assert correlaciones["concept_drift"] < 0.05


def test_el_data_drift_no_degrada_el_modelo_pero_el_concept_drift_si(dataset):
    """El experimento completo, en un solo test.

    Es la afirmación que sostiene todo el proyecto: drift de datos y degradación
    del modelo son cosas distintas. Se entrena con el baseline y se mide el AUC
    en cada régimen.
    """
    baseline = dataset[dataset.regimen == "baseline"]
    X, y = split_features_target(baseline, TARGET)

    modelo = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    modelo.fit(X, y)

    from sklearn.metrics import roc_auc_score

    def auc(regimen: str) -> float:
        subset = dataset[dataset.regimen == regimen]
        X_r, y_r = split_features_target(subset, TARGET)
        return roc_auc_score(y_r, modelo.predict_proba(X_r)[:, 1])

    auc_baseline = auc("baseline")
    auc_data_drift = auc("data_drift")
    auc_concept_drift = auc("concept_drift")

    # Durante el data drift el rendimiento se mantiene: menos de 2 puntos.
    assert abs(auc_data_drift - auc_baseline) < 0.02, (
        f"el data drift no debería degradar el modelo "
        f"(baseline {auc_baseline:.4f} vs data drift {auc_data_drift:.4f})"
    )
    # Con el concept drift cae de forma inequívoca: más de 3 puntos.
    assert auc_baseline - auc_concept_drift > 0.03, (
        f"el concept drift debería degradar el modelo "
        f"(baseline {auc_baseline:.4f} vs concept drift {auc_concept_drift:.4f})"
    )


def test_gender_no_tiene_efecto_sobre_el_churn(dataset):
    """La variable de control. Si diera señal, el generador estaría mal."""
    from scipy import stats

    tabla = pd.crosstab(dataset["gender"], dataset[TARGET])
    _, p, _, _ = stats.chi2_contingency(tabla)
    assert p > 0.01


# --- Pipeline de features -------------------------------------------------


def test_winsorizer_recorta_a_los_limites_aprendidos():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 2))
    X[0, 0] = 1000.0  # outlier brutal

    w = Winsorizer(lower=0.01, upper=0.99).fit(X)
    salida = w.transform(X)

    assert salida[0, 0] < 1000.0
    assert salida[:, 0].max() <= w.upper_bounds_[0]
    assert salida[:, 0].min() >= w.lower_bounds_[0]


def test_winsorizer_aplica_los_limites_de_train_a_datos_nuevos():
    """Los límites se aprenden en `fit` y NO se recalculan en `transform`.

    Recalcularlos sería fuga de información: el preprocesamiento estaría usando
    estadísticos de los datos de evaluación.
    """
    train = np.arange(100, dtype=float).reshape(-1, 1)
    w = Winsorizer(lower=0.0, upper=1.0).fit(train)

    nuevo = np.array([[5000.0]])
    assert w.transform(nuevo)[0, 0] == pytest.approx(train.max())


def test_split_features_target_excluye_identificadores(dataset):
    X, y = split_features_target(dataset, TARGET)
    assert "customer_id" not in X.columns
    assert "snapshot_month" not in X.columns
    assert TARGET not in X.columns
    assert list(X.columns) == FEATURES
    assert len(y) == len(dataset)


def test_el_pipeline_tolera_categorias_no_vistas(dataset):
    """`handle_unknown="ignore"` en acción: una categoría nueva no rompe nada."""
    X, y = split_features_target(dataset.head(2000), TARGET)
    modelo = build_pipeline(LogisticRegression(max_iter=300))
    modelo.fit(X, y)

    nuevo = X.head(1).copy()
    nuevo.loc[:, "region"] = "Región inventada"
    nuevo.loc[:, "payment_method"] = "Criptomoneda"

    proba = modelo.predict_proba(nuevo)[:, 1]
    assert 0.0 <= float(proba[0]) <= 1.0


def test_el_preprocesador_expande_las_categoricas(dataset):
    X, y = split_features_target(dataset.head(1500), TARGET)
    modelo = build_pipeline(LogisticRegression(max_iter=200))
    modelo.fit(X, y)

    transformado = modelo.named_steps["preprocessor"].transform(X.head(5))
    # Más columnas que features de entrada, porque el one-hot las expande.
    assert transformado.shape[1] > len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)

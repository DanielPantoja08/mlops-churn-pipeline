"""Generador del dataset sintético de churn con drift inyectado deliberadamente.

================================================================================
ESTE DOCSTRING ES EL GUION DE LA DEMO. Si solo vas a leer una cosa del repo,
que sea esto.
================================================================================

El problema de casi todos los proyectos de portafolio de MLOps es que usan un
dataset estático: no hay forma de demostrar que el monitoreo *funciona*, porque
nunca pasa nada. Aquí el dataset se genera con tres regímenes conocidos, de modo
que sabemos exactamente qué debe detectar el sistema y cuándo.

18 meses de snapshots mensuales de una base de ~2.500 clientes activos. Cada mes,
los clientes que abandonan salen de la base y entran nuevos; los que sobreviven
acumulan antigüedad. Es decir, la distribución de `tenure_months` evoluciona sola,
sin que se la fuerce.

--------------------------------------------------------------------------------
LOS TRES REGÍMENES
--------------------------------------------------------------------------------

1. BASELINE — 2024-01 a 2024-08 (8 meses)
   Distribuciones estables, relación estable entre variables y churn.
   Tasa de churn ~20 %, que es el desbalance de clases que justifica usar
   `class_weight="balanced"` y mirar PR-AUC, no solo accuracy.
   → Es con estos meses con los que se entrena el modelo en producción.

2. DATA DRIFT — desde 2024-09
   Cambia la DISTRIBUCIÓN de dos variables, en rampa progresiva (no un salto
   brusco, que sería poco realista y demasiado fácil de detectar):
     · `monthly_usage_gb`: media 12 → 20 GB, sd 4 → 7
       (historia de negocio: se lanza un plan de datos ilimitado)
     · `payment_method`: migración de cheque electrónico hacia pagos automáticos
   Los COEFICIENTES no cambian: la relación entre las variables y el churn es
   exactamente la misma.
   → Evidently DEBE marcar drift de datos.
   → El AUC del modelo DEBE mantenerse estable.
   Este contraste es el punto pedagógico central: *drift de datos no implica que
   el modelo se haya degradado*. Un sistema que reentrena con cada alerta de
   drift de datos está quemando dinero.

3. CONCEPT DRIFT — desde 2025-01
   Las distribuciones se mantienen en su nivel ya desplazado, pero cambia la
   RELACIÓN entre las variables y el churn (los coeficientes `beta`):
     · `support_tickets_30d` pasa de ser el predictor más fuerte de churn
       (beta +0.90) a ser prácticamente inocuo (beta -0.25).
       Historia de negocio: se lanza un programa de retención proactiva que
       contacta a quien abre tickets. Ahora quejarse es señal de que te van a
       salvar, no de que te vas.
     · El peso pasa a `contract_type = Month-to-month` combinado con
       `monthly_charges` alto, incluyendo un término de interacción nuevo.
   → El modelo entrenado en el baseline sigue apoyándose en los tickets de
     soporte, que ya no informan. Su AUC CAE.
   → Esto es lo que un sistema de monitoreo tiene que detectar para disparar
     el reentrenamiento.

--------------------------------------------------------------------------------
DETALLES DE IMPLEMENTACIÓN QUE IMPORTAN
--------------------------------------------------------------------------------

· `churn` no se asigna con reglas if/else, sino con un modelo logístico latente:
  p = sigmoid(beta_0 + sum(beta_i * x_i)). Eso hace que la relación sea ruidosa
  y aprendible, no determinista, y que el "concept drift" sea literalmente un
  cambio de coeficientes: explícito, medible y explicable.

· El intercepto se CALIBRA por bisección cada mes contra la población real de
  ese mes, para fijar la tasa base de churn en ~20 %. Sin esta calibración la
  tasa cae del 22 % al 7 % a lo largo de los 18 meses (el uso al alza y la
  antigüedad acumulada de los supervivientes tiran de ella hacia abajo), y
  entonces sería imposible distinguir "el modelo se degradó" de "ahora se va
  menos gente". Al fijarla, el concept drift afecta solo al ORDENAMIENTO de
  clientes por riesgo — justo lo que mide el AUC.

· `gender` se genera SIN ningún efecto sobre el churn, a propósito. Es la
  variable de control del EDA: el test de chi-cuadrado de la Fase 2 debe salir
  no significativo. Un EDA que "encuentra" señal en todas las variables es un
  EDA que no está midiendo nada.

· Semilla fija (42). `python src/data/generate_data.py` produce byte a byte el
  mismo dataset en cualquier máquina.

Uso:
    python src/data/generate_data.py
    python src/data/generate_data.py --customers 5000 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import RAW_DATA_DIR  # noqa: E402

# --------------------------------------------------------------------------
# Calendario y regímenes
# --------------------------------------------------------------------------

START_MONTH = "2024-01"
N_MONTHS = 18

BASELINE_MONTHS = 8  # índices 0-7  → 2024-01 .. 2024-08
DATA_DRIFT_START = 8  # índice 8    → 2024-09
CONCEPT_DRIFT_START = 12  # índice 12 → 2025-01
DATA_DRIFT_RAMP = 4  # meses que tarda la rampa en completarse

DEFAULT_ACTIVE_CUSTOMERS = 2500
DEFAULT_SEED = 42
TARGET_CHURN_RATE = 0.20

# --------------------------------------------------------------------------
# Categorías
# --------------------------------------------------------------------------

REGIONS = ["Norte", "Centro", "Sur", "Occidente"]
REGION_PROBS = [0.22, 0.34, 0.26, 0.18]

CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
CONTRACT_PROBS = [0.52, 0.28, 0.20]

PAYMENT_METHODS = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
PAYMENT_PROBS_BASELINE = [0.35, 0.20, 0.22, 0.23]
# Tras el drift: migración a pagos automáticos (menos cheques, más tarjeta).
PAYMENT_PROBS_DRIFTED = [0.20, 0.10, 0.30, 0.40]

# --------------------------------------------------------------------------
# Estadísticos de referencia para estandarizar el predictor lineal.
#
# Son CONSTANTES fijas del baseline, no se recalculan por mes. Es deliberado:
# si se recalcularan, un desplazamiento de la distribución se "absorbería" en la
# estandarización y el drift de datos se volvería invisible para el modelo
# latente. Al fijarlas, un cambio de distribución realmente mueve el predictor.
# --------------------------------------------------------------------------

REFERENCE_STATS: dict[str, tuple[float, float]] = {
    "age": (45.0, 15.0),
    "tenure_months": (24.0, 18.0),
    "monthly_charges": (70.0, 25.0),
    "monthly_usage_gb": (12.0, 4.0),
    "support_tickets_30d": (0.8, 1.0),
    "avg_session_minutes": (25.0, 10.0),
    "num_services": (3.0, 1.4),
    "late_payments_3m": (0.4, 0.8),
}


@dataclass
class Betas:
    """Coeficientes del modelo logístico latente que decide el churn.

    Cambiar este objeto ES el concept drift. No hay ninguna otra magia.
    """

    numeric: dict[str, float]
    contract: dict[str, float]
    payment: dict[str, float]
    senior: float
    paperless: float
    region: dict[str, float]
    # Interacción Month-to-month x monthly_charges (estandarizado).
    interaction_mtm_charges: float = 0.0
    intercept: float = 0.0


BETAS_BASELINE = Betas(
    numeric={
        "age": 0.05,
        "tenure_months": -0.85,
        "monthly_charges": 0.45,
        "monthly_usage_gb": -0.35,
        "support_tickets_30d": 0.90,  # el predictor más fuerte del baseline
        "avg_session_minutes": -0.20,
        "num_services": -0.30,
        "late_payments_3m": 0.55,
    },
    contract={"Month-to-month": 0.80, "One year": -0.35, "Two year": -0.95},
    payment={
        "Electronic check": 0.40,
        "Mailed check": 0.10,
        "Bank transfer": -0.15,
        "Credit card": -0.20,
    },
    senior=0.20,
    paperless=0.15,
    region={"Norte": 0.05, "Centro": -0.05, "Sur": 0.10, "Occidente": -0.08},
    interaction_mtm_charges=0.0,
)

# Concept drift: mismo esquema de variables, otra relación con el objetivo.
BETAS_CONCEPT_DRIFT = Betas(
    numeric={
        "age": 0.05,
        "tenure_months": -0.70,
        "monthly_charges": 0.95,  # 0.45 -> 0.95, el precio pasa a dominar
        "monthly_usage_gb": -0.30,
        "support_tickets_30d": -0.25,  # 0.90 -> -0.25, SE INVIERTE
        "avg_session_minutes": -0.20,
        "num_services": -0.25,
        "late_payments_3m": 0.30,  # 0.55 -> 0.30, pierde fuerza
    },
    contract={"Month-to-month": 1.40, "One year": -0.40, "Two year": -1.10},
    payment={
        "Electronic check": 0.25,
        "Mailed check": 0.10,
        "Bank transfer": -0.10,
        "Credit card": -0.15,
    },
    senior=0.20,
    paperless=0.10,
    region={"Norte": 0.05, "Centro": -0.05, "Sur": 0.10, "Occidente": -0.08},
    interaction_mtm_charges=0.60,  # término nuevo que el modelo viejo no conoce
)


@dataclass
class MonthlyDistribution:
    """Parámetros de las distribuciones de un mes concreto."""

    usage_mean: float
    usage_sd: float
    payment_probs: list[float] = field(default_factory=lambda: list(PAYMENT_PROBS_BASELINE))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _standardize(values: np.ndarray, column: str) -> np.ndarray:
    mean, sd = REFERENCE_STATS[column]
    return (values - mean) / sd


def month_labels(start: str = START_MONTH, n_months: int = N_MONTHS) -> list[str]:
    """['2024-01', '2024-02', ...] usando aritmética de periodos de pandas."""
    return [str(p) for p in pd.period_range(start=start, periods=n_months, freq="M")]


def regime_for(month_index: int) -> str:
    """Nombre del régimen activo en un índice de mes dado."""
    if month_index >= CONCEPT_DRIFT_START:
        return "concept_drift"
    if month_index >= DATA_DRIFT_START:
        return "data_drift"
    return "baseline"


def distribution_for(month_index: int) -> MonthlyDistribution:
    """Distribuciones del mes, aplicando la rampa de data drift si corresponde.

    La rampa evita un salto brusco: el drift real en producción casi nunca es
    un escalón, y un escalón haría el problema de detección trivial.
    """
    if month_index < DATA_DRIFT_START:
        return MonthlyDistribution(usage_mean=12.0, usage_sd=4.0)

    # progreso de 0 a 1 a lo largo de DATA_DRIFT_RAMP meses, luego se estabiliza
    steps_in = month_index - DATA_DRIFT_START + 1
    progress = min(steps_in / DATA_DRIFT_RAMP, 1.0)

    usage_mean = 12.0 + progress * (20.0 - 12.0)
    usage_sd = 4.0 + progress * (7.0 - 4.0)
    payment_probs = [
        base + progress * (drifted - base)
        for base, drifted in zip(PAYMENT_PROBS_BASELINE, PAYMENT_PROBS_DRIFTED, strict=True)
    ]
    total = sum(payment_probs)
    payment_probs = [p / total for p in payment_probs]

    return MonthlyDistribution(usage_mean=usage_mean, usage_sd=usage_sd, payment_probs=payment_probs)


def betas_for(month_index: int) -> Betas:
    """Coeficientes activos. Cambiarlos en 2025-01 ES el concept drift."""
    if month_index >= CONCEPT_DRIFT_START:
        return BETAS_CONCEPT_DRIFT
    return BETAS_BASELINE


# --------------------------------------------------------------------------
# Generación de clientes
# --------------------------------------------------------------------------


def _new_customers(
    rng: np.random.Generator,
    n: int,
    next_id: int,
    dist: MonthlyDistribution,
    is_initial_cohort: bool,
) -> pd.DataFrame:
    """Crea clientes nuevos con sus atributos estables (los que no cambian mes a mes)."""
    contract = rng.choice(CONTRACT_TYPES, size=n, p=CONTRACT_PROBS)

    # El cargo mensual depende del contrato: los contratos largos salen más
    # baratos por mes. Introducir esta correlación es importante para que el
    # EDA bivariado tenga algo real que encontrar.
    contract_discount = np.select(
        [contract == "One year", contract == "Two year"], [-8.0, -14.0], default=0.0
    )
    monthly_charges = np.clip(
        rng.normal(70.0, 25.0, size=n) + contract_discount, 18.0, 190.0
    ).round(2)

    if is_initial_cohort:
        # La base inicial ya tiene historia: antigüedades repartidas.
        tenure = rng.integers(0, 60, size=n)
    else:
        # Los clientes que entran después empiezan casi desde cero.
        tenure = rng.integers(0, 3, size=n)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(next_id, next_id + n)],
            "age": np.clip(rng.normal(45.0, 15.0, size=n), 18, 92).round().astype(int),
            "gender": rng.choice(["F", "M"], size=n, p=[0.49, 0.51]),
            "region": rng.choice(REGIONS, size=n, p=REGION_PROBS),
            "senior_citizen": np.where(rng.random(n) < 0.16, "Yes", "No"),
            "contract_type": contract,
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "payment_method": rng.choice(PAYMENT_METHODS, size=n, p=dist.payment_probs),
            "paperless_billing": np.where(rng.random(n) < 0.58, "Yes", "No"),
            "num_services": np.clip(rng.poisson(2.6, size=n) + 1, 1, 8),
        }
    )


def _refresh_behaviour(
    rng: np.random.Generator, customers: pd.DataFrame, dist: MonthlyDistribution
) -> pd.DataFrame:
    """Vuelve a muestrear las variables de comportamiento del mes.

    Estas son las que se mueven en el régimen de data drift. Las variables
    contractuales y demográficas son estables por cliente.
    """
    n = len(customers)
    customers = customers.copy()

    # Uso: lognormal reescalada para que tenga cola derecha (hay clientes que
    # consumen muchísimo). El EDA univariado debe detectar esos outliers.
    raw_usage = rng.lognormal(mean=0.0, sigma=0.45, size=n)
    raw_usage = raw_usage / raw_usage.mean()
    customers["monthly_usage_gb"] = np.clip(
        raw_usage * dist.usage_mean + rng.normal(0.0, dist.usage_sd * 0.25, size=n), 0.1, None
    ).round(2)

    # Los clientes con más servicios abren más tickets: correlación intencionada.
    ticket_lambda = 0.45 + 0.12 * customers["num_services"].to_numpy()
    customers["support_tickets_30d"] = rng.poisson(ticket_lambda)

    customers["avg_session_minutes"] = np.clip(
        rng.normal(25.0, 10.0, size=n) + 0.35 * customers["monthly_usage_gb"].to_numpy(), 1.0, None
    ).round(1)

    # Los impagos se concentran en quien paga con cheque electrónico.
    late_lambda = np.where(
        customers["payment_method"].to_numpy() == "Electronic check", 0.75, 0.28
    )
    customers["late_payments_3m"] = rng.poisson(late_lambda)

    return customers


def _linear_predictor(df: pd.DataFrame, betas: Betas) -> np.ndarray:
    """Predictor lineal del modelo latente que genera el churn."""
    z = np.full(len(df), betas.intercept, dtype=float)

    for column, beta in betas.numeric.items():
        z += beta * _standardize(df[column].to_numpy(dtype=float), column)

    z += df["contract_type"].map(betas.contract).to_numpy(dtype=float)
    z += df["payment_method"].map(betas.payment).to_numpy(dtype=float)
    z += df["region"].map(betas.region).to_numpy(dtype=float)
    z += np.where(df["senior_citizen"].to_numpy() == "Yes", betas.senior, 0.0)
    z += np.where(df["paperless_billing"].to_numpy() == "Yes", betas.paperless, 0.0)

    if betas.interaction_mtm_charges:
        is_mtm = (df["contract_type"].to_numpy() == "Month-to-month").astype(float)
        charges_std = _standardize(df["monthly_charges"].to_numpy(dtype=float), "monthly_charges")
        z += betas.interaction_mtm_charges * is_mtm * charges_std

    # `gender` no aparece por ningún lado: es la variable de control del EDA.
    return z


def _solve_intercept(z_without_intercept: np.ndarray, target_rate: float) -> float:
    """Intercepto que hace que la tasa media de churn del mes sea `target_rate`.

    POR QUÉ ESTO ES NECESARIO (y no un truco para maquillar los datos):

    Sin calibrar, la tasa base de churn se movería sola por dos motivos que no
    tienen nada que ver con lo que queremos demostrar:

      1. El data drift sube `monthly_usage_gb`, que tiene coeficiente negativo,
         así que la tasa de churn bajaría por pura aritmética.
      2. Los supervivientes acumulan antigüedad mes a mes y `tenure_months`
         también tiene coeficiente negativo: la base se vuelve estructuralmente
         más leal con el tiempo.

    Con ambos efectos sueltos, la tasa cae del 22 % al 7 % en 18 meses, y
    entonces cualquier caída de rendimiento del modelo es inseparable de "es que
    ahora se va mucha menos gente". Fijando la tasa base, el concept drift
    afecta solo al ORDENAMIENTO de clientes por riesgo, que es exactamente lo
    que mide el AUC. Es la única forma de que el experimento tenga una variable
    independiente limpia.

    Bisección: la tasa media es monótona creciente en el intercepto.
    """
    low, high = -15.0, 15.0
    for _ in range(60):
        mid = (low + high) / 2
        if _sigmoid(z_without_intercept + mid).mean() < target_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def generate_dataset(
    n_active: int = DEFAULT_ACTIVE_CUSTOMERS,
    seed: int = DEFAULT_SEED,
    n_months: int = N_MONTHS,
) -> dict[str, pd.DataFrame]:
    """Genera todos los meses. Devuelve {'2024-01': DataFrame, ...}."""
    rng = np.random.default_rng(seed)
    labels = month_labels(n_months=n_months)
    monthly: dict[str, pd.DataFrame] = {}

    active: pd.DataFrame | None = None
    next_id = 1

    for idx, label in enumerate(labels):
        dist = distribution_for(idx)
        betas = betas_for(idx)

        if active is None:
            active = _new_customers(rng, n_active, next_id, dist, is_initial_cohort=True)
            next_id += n_active
        else:
            # Reponer la base: entran clientes nuevos donde hubo bajas, más un
            # ligero crecimiento neto del 1 %.
            target = int(n_active * (1.0 + 0.01 * idx))
            missing = max(target - len(active), 0)
            if missing:
                incoming = _new_customers(rng, missing, next_id, dist, is_initial_cohort=False)
                next_id += missing
                active = pd.concat([active, incoming], ignore_index=True)
            # Algunos clientes cambian de método de pago siguiendo la tendencia
            # del mes: así el drift de `payment_method` alcanza también a la
            # base existente, no solo a los clientes nuevos.
            switching = rng.random(len(active)) < 0.05
            if switching.any():
                active.loc[switching, "payment_method"] = rng.choice(
                    PAYMENT_METHODS, size=int(switching.sum()), p=dist.payment_probs
                )

        snapshot = _refresh_behaviour(rng, active, dist)

        # El intercepto se resuelve contra la población real de ESTE mes, para
        # que la tasa base de churn quede fijada y no confunda el experimento.
        z0 = _linear_predictor(snapshot, betas)
        probabilities = _sigmoid(z0 + _solve_intercept(z0, TARGET_CHURN_RATE))
        snapshot["churn"] = (rng.random(len(snapshot)) < probabilities).astype(int)
        snapshot.insert(1, "snapshot_month", label)

        column_order = [
            "customer_id",
            "snapshot_month",
            "age",
            "gender",
            "region",
            "senior_citizen",
            "contract_type",
            "tenure_months",
            "monthly_charges",
            "payment_method",
            "paperless_billing",
            "monthly_usage_gb",
            "support_tickets_30d",
            "avg_session_minutes",
            "num_services",
            "late_payments_3m",
            "churn",
        ]
        monthly[label] = snapshot[column_order]

        # Los que abandonaron salen de la base; los que siguen suman un mes.
        survivors = active.loc[snapshot["churn"].to_numpy() == 0].copy()
        survivors["tenure_months"] = survivors["tenure_months"] + 1
        active = survivors.reset_index(drop=True)

    return monthly


def write_dataset(monthly: dict[str, pd.DataFrame], output_dir: Path = RAW_DATA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, frame in monthly.items():
        frame.to_csv(output_dir / f"{label}.csv", index=False)


def load_raw(months: list[str] | None = None, data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Carga uno, varios o todos los meses en un único DataFrame.

    Utilidad compartida por el EDA, el entrenamiento y el monitoreo, para que
    los tres lean los datos exactamente igual.
    """
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No hay CSVs en {data_dir}. Ejecuta primero: python src/data/generate_data.py"
        )
    if months is not None:
        wanted = set(months)
        paths = [p for p in paths if p.stem in wanted]
        missing = wanted - {p.stem for p in paths}
        if missing:
            raise FileNotFoundError(f"Faltan los meses: {sorted(missing)}")
    return pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)


def _summary(monthly: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for idx, (label, frame) in enumerate(monthly.items()):
        rows.append(
            {
                "mes": label,
                "regimen": regime_for(idx),
                "clientes": len(frame),
                "churn_rate": round(frame["churn"].mean(), 4),
                "uso_medio_gb": round(frame["monthly_usage_gb"].mean(), 2),
                "corr_tickets_churn": round(
                    frame["support_tickets_30d"].corr(frame["churn"]), 4
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--customers", type=int, default=DEFAULT_ACTIVE_CUSTOMERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--months", type=int, default=N_MONTHS)
    parser.add_argument("--output", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args()

    monthly = generate_dataset(n_active=args.customers, seed=args.seed, n_months=args.months)
    write_dataset(monthly, args.output)

    summary = _summary(monthly)
    print(f"\nGenerados {len(monthly)} meses en {args.output}\n")
    print(summary.to_string(index=False))
    print(
        "\nLectura esperada:"
        "\n  · 'uso_medio_gb' sube desde 2024-09  -> data drift"
        "\n  · 'corr_tickets_churn' se derrumba en 2025-01 -> concept drift"
        "\n  · 'churn_rate' se mantiene estable -> el drift no es un cambio de tasa base\n"
    )


if __name__ == "__main__":
    main()

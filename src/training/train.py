"""Entrenamiento de los modelos de churn, con seguimiento en MLflow.

Entrena tres modelos sobre el mismo pipeline de features, los compara con el
mismo criterio y registra el ganador con el alias `champion`.

    python src/training/train.py

--------------------------------------------------------------------------------
POR QUÉ EL SPLIT ES TEMPORAL Y NO ALEATORIO
--------------------------------------------------------------------------------
Es la decisión que más cambia los resultados, y la que más se hace mal.

Un `train_test_split(shuffle=True)` sobre este dataset mezclaría los tres
regímenes, y además pondría al mismo cliente en train y en validación (el
dataset es un panel: cada cliente aparece en varios meses). El resultado sería
una métrica de validación optimista que no se parecería en nada al rendimiento
real, porque el escenario que mide — predecir un mes con información de meses
posteriores — no existe en producción.

Aquí se entrena con 2024-01..06 y se valida con 2024-07..08: pasado para
aprender, futuro inmediato para evaluar. Exactamente la situación real.

--------------------------------------------------------------------------------
POR QUÉ ESTAS MÉTRICAS
--------------------------------------------------------------------------------
Con un desbalance de 80/20 (notebook 01), la accuracy es inútil: predecir
siempre "permanece" da un 80 %.

  · AUC-ROC     — criterio de selección. Mide la capacidad de ORDENAR clientes
                  por riesgo, que es lo que se usa en la práctica (a quién llama
                  primero el equipo de retención). Es además independiente del
                  umbral y de la prevalencia, lo que la hace comparable entre
                  meses en la Fase 8.
  · PR-AUC      — más exigente sobre la clase minoritaria; sube y baja con el
                  desbalance donde la AUC-ROC no se inmuta.
  · Precision / recall / F1 — al umbral 0.5, para dar una lectura operativa.
  · Brier score — calibración. Un modelo puede ordenar bien y aun así dar
                  probabilidades mal calibradas; si esas probabilidades se van a
                  usar para estimar valor esperado, la calibración importa.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin backend interactivo: esto corre también en CI

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import tracking  # noqa: E402
from src.config import (  # noqa: E402
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    TARGET,
    TRAIN_MONTHS,
    VALIDATION_MONTHS,
)
from src.data.generate_data import load_raw  # noqa: E402
from src.features.build_features import (  # noqa: E402
    TRUSTED_TYPES,
    build_pipeline,
    feature_names,
    split_features_target,
)
from src.viz import SERIES, STATUS, set_style  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42


def define_models(scale_pos_weight: float) -> dict[str, object]:
    """Los tres candidatos.

    `class_weight="balanced"` y `scale_pos_weight` son la misma idea aplicada a
    cada familia: penalizar más el error sobre la clase minoritaria, para que el
    modelo no se limite a predecir "permanece" siempre (notebook 01).
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_proba),
        "positive_rate": float(y_pred.mean()),
    }


def _curves_figure(y_true: np.ndarray, y_proba: np.ndarray, name: str):
    """Curvas ROC y precision-recall, una al lado de la otra."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.4))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax_roc.plot(fpr, tpr, color=SERIES[0], label=f"AUC = {roc_auc_score(y_true, y_proba):.4f}")
    ax_roc.plot([0, 1], [0, 1], color="#c3c2b7", linestyle="--", linewidth=1.2, label="Azar")
    ax_roc.set_xlabel("Tasa de falsos positivos")
    ax_roc.set_ylabel("Tasa de verdaderos positivos")
    ax_roc.set_title("Curva ROC")
    ax_roc.legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    base = y_true.mean()
    ax_pr.plot(
        recall,
        precision,
        color=SERIES[1],
        label=f"PR-AUC = {average_precision_score(y_true, y_proba):.4f}",
    )
    ax_pr.axhline(base, color="#c3c2b7", linestyle="--", linewidth=1.2, label=f"Base = {base:.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Curva precision-recall")
    ax_pr.legend(loc="upper right")

    fig.suptitle(name, x=0.02, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _confusion_figure(y_true: np.ndarray, y_proba: np.ndarray, name: str):
    matriz = confusion_matrix(y_true, (y_proba >= 0.5).astype(int))
    etiquetas = ["Permanece", "Abandona"]

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(matriz, cmap="Blues")
    ax.set_xticks([0, 1], etiquetas)
    ax.set_yticks([0, 1], etiquetas)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.grid(False)
    umbral = matriz.max() / 2
    for i in range(2):
        for j in range(2):
            ax.annotate(
                f"{matriz[i, j]:,}",
                xy=(j, i),
                ha="center",
                va="center",
                fontsize=13,
                color="#ffffff" if matriz[i, j] > umbral else "#0b0b0b",
            )
    ax.set_title(f"Matriz de confusión · {name}", fontsize=11)
    fig.tight_layout()
    return fig


def _importance_figure(pipeline, name: str):
    """Importancia de features, con el signo cuando el modelo lo tiene.

    Sirve de control de cordura: `gender` no debería aparecer arriba. El EDA
    (notebook 03) mostró que no tiene asociación con el churn, así que un modelo
    que le dé mucho peso está sobreajustando.
    """
    nombres = feature_names(pipeline)
    modelo = pipeline.named_steps["model"]

    if hasattr(modelo, "coef_"):
        valores = modelo.coef_[0]
        titulo = "Coeficientes (log-odds)"
    elif hasattr(modelo, "feature_importances_"):
        valores = modelo.feature_importances_
        titulo = "Importancia de features"
    else:
        return None

    serie = pd.Series(valores, index=nombres)
    top = serie.reindex(serie.abs().sort_values(ascending=False).index).head(15).iloc[::-1]
    colores = [STATUS["critical"] if v > 0 else SERIES[0] for v in top.to_numpy()]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.index, top.to_numpy(), color=colores, height=0.65)
    ax.axvline(0, color="#c3c2b7", linewidth=1.2)
    ax.set_title(f"{titulo} · {name}")
    ax.grid(False)
    fig.tight_layout()
    return fig


def train_one(
    algorithm: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[str, dict[str, float]]:
    """Entrena, evalúa y registra un run. Devuelve (run_id, métricas)."""
    with mlflow.start_run(run_name=algorithm) as run:
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        proba_train = pipeline.predict_proba(X_train)[:, 1]
        proba_val = pipeline.predict_proba(X_val)[:, 1]

        metricas_train = evaluate(y_train.to_numpy(), proba_train)
        metricas_val = evaluate(y_val.to_numpy(), proba_val)

        mlflow.log_params(
            {
                "algorithm": algorithm,
                "train_months": ",".join(TRAIN_MONTHS),
                "validation_months": ",".join(VALIDATION_MONTHS),
                "n_train": len(X_train),
                "n_validation": len(X_val),
                "train_churn_rate": round(float(y_train.mean()), 4),
                "split_strategy": "temporal",
                **{f"model__{k}": v for k, v in estimator.get_params().items() if v is not None},
            }
        )
        mlflow.log_metrics({f"train_{k}": v for k, v in metricas_train.items()})
        mlflow.log_metrics(metricas_val)

        # La brecha train-validación es la señal de sobreajuste más directa.
        mlflow.log_metric("overfit_gap_auc", metricas_train["roc_auc"] - metricas_val["roc_auc"])

        for figura, nombre in [
            (_curves_figure(y_val.to_numpy(), proba_val, algorithm), "curvas"),
            (_confusion_figure(y_val.to_numpy(), proba_val, algorithm), "matriz_confusion"),
            (_importance_figure(pipeline, algorithm), "importancia_features"),
        ]:
            if figura is not None:
                mlflow.log_figure(figura, f"figuras/{nombre}.png")
                plt.close(figura)

        # Dos detalles necesarios para que el pipeline con el `Winsorizer`
        # personalizado se pueda guardar y volver a cargar:
        #
        # · `code_paths` empaqueta el código del proyecto junto al modelo. Sin
        #   esto, cargarlo en otro entorno (el contenedor de la API) fallaría al
        #   no poder importar la clase.
        # · `skops_trusted_types`: MLflow 3 serializa los modelos de sklearn con
        #   skops en vez de pickle, que por seguridad se niega a cargar tipos que
        #   no estén declarados explícitamente como confiables. Es un cambio
        #   bienvenido — un pickle es ejecución de código arbitraria — y la
        #   respuesta correcta es declarar los tipos propios, no volver a pickle.
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=infer_signature(X_val, proba_val),
            input_example=X_val.head(3),
            code_paths=["src"],
            skops_trusted_types=TRUSTED_TYPES,
        )

        print(
            f"  {algorithm:>20}  AUC {metricas_val['roc_auc']:.4f}"
            f"  PR-AUC {metricas_val['pr_auc']:.4f}"
            f"  F1 {metricas_val['f1']:.4f}"
            f"  brecha {metricas_train['roc_auc'] - metricas_val['roc_auc']:+.4f}"
        )
        return run.info.run_id, metricas_val


def register_champion(run_id: str, algorithm: str, metrics: dict[str, float]) -> str:
    """Registra el modelo ganador y le pone el alias `champion`."""
    client = MlflowClient()
    version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=REGISTERED_MODEL_NAME,
    )
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "algorithm", algorithm)
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, version.version, "validation_roc_auc", f"{metrics['roc_auc']:.4f}"
    )
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version.version,
        description=(
            f"{algorithm} entrenado con {TRAIN_MONTHS[0]}..{TRAIN_MONTHS[-1]} "
            f"y validado con {VALIDATION_MONTHS[0]}..{VALIDATION_MONTHS[-1]}. "
            f"AUC de validación {metrics['roc_auc']:.4f}."
        ),
    )
    return version.version


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena y registra el modelo de churn")
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="entrena y compara, pero no toca el Model Registry",
    )
    args = parser.parse_args()

    set_style()
    tracking.setup()
    print(f"MLflow tracking: {mlflow.get_tracking_uri()}")
    print(f"Artefactos:      {tracking.artifact_location() or 'ruta local por defecto'}\n")

    df = load_raw(months=TRAIN_MONTHS + VALIDATION_MONTHS)
    train_df = df[df.snapshot_month.isin(TRAIN_MONTHS)]
    val_df = df[df.snapshot_month.isin(VALIDATION_MONTHS)]

    X_train, y_train = split_features_target(train_df, TARGET)
    X_val, y_val = split_features_target(val_df, TARGET)

    print(f"Train      {TRAIN_MONTHS[0]}..{TRAIN_MONTHS[-1]}  {len(X_train):>6,} filas  churn {y_train.mean():.2%}")
    print(f"Validación {VALIDATION_MONTHS[0]}..{VALIDATION_MONTHS[-1]}  {len(X_val):>6,} filas  churn {y_val.mean():.2%}\n")

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    resultados: dict[str, tuple[str, dict[str, float]]] = {}

    for algorithm, estimator in define_models(scale_pos_weight).items():
        run_id, metricas = train_one(algorithm, estimator, X_train, y_train, X_val, y_val)
        resultados[algorithm] = (run_id, metricas)

    comparativa = (
        pd.DataFrame({a: m for a, (_, m) in resultados.items()}).T.sort_values(
            "roc_auc", ascending=False
        )
    ).round(4)
    print("\nComparativa sobre el conjunto de validación:")
    print(comparativa.to_string())

    ganador = comparativa.index[0]
    run_id, metricas = resultados[ganador]

    if args.no_register:
        print(f"\nGanador: {ganador} (no se registra, --no-register)")
        return

    version = register_champion(run_id, ganador, metricas)
    print(
        f"\nGanador: {ganador} · registrado como {REGISTERED_MODEL_NAME} "
        f"v{version} con alias @{CHAMPION_ALIAS}"
    )
    print(f"La API lo resolverá como: models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")

    resumen = {
        "champion": ganador,
        "version": version,
        "run_id": run_id,
        "metrics": {k: round(v, 4) for k, v in metricas.items()},
    }
    print(f"\n{json.dumps(resumen, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

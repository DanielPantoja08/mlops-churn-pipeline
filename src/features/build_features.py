"""Pipeline de transformación de features.

Implementa, una por una, las decisiones documentadas al cierre de los notebooks
de EDA de la Fase 2. Cada transformación tiene abajo el hallazgo que la motiva;
si el EDA no lo justificó, no está aquí.

--------------------------------------------------------------------------------
LA DECISIÓN DE DISEÑO MÁS IMPORTANTE DEL PROYECTO
--------------------------------------------------------------------------------
El preprocesamiento vive DENTRO de un `Pipeline` de sklearn que se serializa
junto al modelo, no en un script aparte que haya que reproducir al servir.

Es la causa número uno de errores silenciosos en producción — el
*training/serving skew*: el modelo se entrenó con datos escalados de una forma
y en producción recibe datos escalados de otra, o sin escalar, y sigue
devolviendo predicciones con toda normalidad. No hay excepción, no hay error en
los logs; simplemente las predicciones son peores y nadie se entera.

Con el preprocesamiento dentro del pipeline, la API de la Fase 4 recibe un JSON
con los valores crudos del cliente y llama a `predict_proba`. No sabe nada de
escalado, ni de one-hot, ni de winsorización, y no puede desincronizarse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

# Tipos que MLflow debe aceptar al deserializar el pipeline.
#
# MLflow 3 guarda los modelos de sklearn con `skops` en lugar de `pickle`, y
# skops se niega por defecto a reconstruir cualquier tipo que no esté declarado
# como confiable — un pickle, al cargarse, ejecuta código arbitrario, así que el
# cambio es acertado. La respuesta correcta es declarar los tipos propios aquí,
# no volver a pickle. Esta lista vive junto a la clase para que no se olvide
# actualizarla si el pipeline incorpora otro transformador propio.
TRUSTED_TYPES = [
    "numpy.dtype",
    "src.features.build_features.Winsorizer",
    # Los estimadores de gradient boosting tampoco vienen en la lista blanca de
    # skops, porque no forman parte de sklearn.
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "collections.OrderedDict",  # LightGBM lo usa internamente
]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Recorta cada columna a sus percentiles [lower, upper] aprendidos en `fit`.

    MOTIVACIÓN (notebook 02): `monthly_usage_gb` tiene una cola derecha larga —
    el máximo está muy por encima del percentil 99. Son clientes reales que
    consumen mucho, no errores de medición, así que eliminar esas filas sería
    tirar información válida. Pero dejarlas sin tratar distorsiona la regresión
    logística, que es sensible a los valores extremos.

    Recortar en vez de eliminar conserva la fila entera y solo limita hasta dónde
    puede llegar el valor extremo.

    Los límites se aprenden SOLO con los datos de entrenamiento. Calcularlos
    sobre el dataset completo sería fuga de información: el modelo estaría
    usando, indirectamente, estadísticos del futuro.

    En producción es además una red de seguridad: si llega un valor absurdo por
    un fallo del sistema origen, queda acotado al rango visto en entrenamiento
    en vez de propagarse a la predicción.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):  # noqa: N803
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):  # noqa: N803
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Preprocesador de columnas numéricas y categóricas.

    Numéricas — imputación por mediana, winsorización p1–p99, estandarización:
      · La imputación no hace falta con el dataset actual (notebook 01: no hay
        nulos), pero se incluye porque en producción sí puede faltar un campo, y
        un servicio que revienta ante un nulo no es un servicio.
      · La mediana y no la media, precisamente por la cola derecha del uso.
      · `StandardScaler` porque las escalas son muy dispares (notebook 02): sin
        él la regresión logística quedaría dominada por `monthly_charges`, que
        se mide en decenas, frente a `support_tickets_30d`, que vale 0, 1 o 2.

    Categóricas — imputación por moda y one-hot:
      · Cardinalidad baja, entre 2 y 4 niveles, sin categorías raras que
        agrupar (notebook 02).
      · `handle_unknown="ignore"` no es cosmético: en producción puede llegar
        una categoría que no existía al entrenar (una región nueva, un método de
        pago nuevo). Sin esta opción la API devolvería un 500; con ella, la
        codifica como todo ceros y sigue sirviendo.
    """
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("winsorizer", Winsorizer(lower=0.01, upper=0.99)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # descarta customer_id, snapshot_month y todo lo demás
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator) -> Pipeline:
    """Une el preprocesador con un estimador en un único objeto serializable."""
    return Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])


def split_features_target(df: pd.DataFrame, target: str = "churn"):
    """Separa X e y usando solo las columnas declaradas como features.

    `customer_id` se excluye deliberadamente (notebook 01): es un identificador,
    y como el dataset es un panel donde el mismo cliente aparece en varios meses,
    incluirlo permitiría memorizar clientes concretos y provocaría fuga entre
    entrenamiento y validación.
    """
    return df[FEATURES].copy(), df[target].copy()


def feature_names(pipeline: Pipeline) -> list[str]:
    """Nombres de las features tras el preprocesamiento, ya expandido el one-hot.

    Necesario para poder interpretar coeficientes e importancias: sin esto, la
    importancia de features es una lista de números sin etiqueta.
    """
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())

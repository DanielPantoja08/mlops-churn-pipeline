# MLOps Churn Pipeline

Pipeline MLOps end-to-end para predicción de abandono de clientes (*churn*).

El objetivo de este proyecto no es entrenar un modelo con buen AUC — eso es la parte fácil.
El objetivo es demostrar el **ciclo de vida completo de un modelo en producción**: explorarlo con
criterio, versionar datos y experimentos, desplegarlo como servicio, **detectar automáticamente
cuándo se degrada** y tener el camino listo para reentrenarlo.

Para que ese último punto sea demostrable y no teórico, el dataset es sintético y tiene **drift
inyectado deliberadamente**: sé exactamente en qué mes empieza, de qué tipo es y por qué. Eso
permite comprobar que el sistema de monitoreo detecta lo que tiene que detectar.

## Stack

Python · DVC · MLflow · FastAPI · Docker · Evidently AI · Streamlit · GitHub Actions ·
[Floci](https://floci.io) (emulador de GCP en contenedor)

## Estado

🚧 En construcción. Este README se amplía en cada fase; la versión final incluye el diagrama de
arquitectura, los hallazgos del EDA y las decisiones de diseño.

El plan completo por fases está en [`docs/roadmap.md`](docs/roadmap.md).

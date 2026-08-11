# Guion de demo · 2 minutos

Pensado para una entrevista técnica. La idea es que **no se enseñe el código**: se enseña el
sistema funcionando y se explica *por qué* está montado así. El código sale solo si preguntan.

## Preparación (antes de empezar, no durante)

```bash
docker compose up -d --build
uv run python scripts/bootstrap_floci.py
MLFLOW_TRACKING_URI=http://localhost:5000 STORAGE_EMULATOR_HOST=http://localhost:4588 \
  uv run python src/training/train.py
curl -X POST http://localhost:8000/reload
uv run python monitoring/generate_report.py
docker compose --profile tools up -d dashboard
```

Pestañas abiertas: dashboard (`:8501`), MLflow (`:5000`), API docs (`:8000/docs`).

---

## 0:00 – 0:20 · El problema, no el modelo

> «Este es un pipeline de predicción de churn, pero lo interesante no es el modelo. El modelo es
> una regresión logística con un AUC de 0,86. Lo interesante es que **sé exactamente cuándo va a
> dejar de funcionar**, porque el dataset tiene drift inyectado en fechas conocidas.»

Abrir el dashboard. Señalar el semáforo rojo de la cabecera.

---

## 0:20 – 0:55 · Las dos señales

Pestaña **Drift**. Los dos gráficos apilados.

> «Arriba, el porcentaje de columnas cuya distribución ha cambiado respecto a los datos de
> entrenamiento. Está disponible en tiempo real y sin etiquetas.
>
> Abajo, el AUC real. Es la verdad, pero en producción llegaría con semanas de retraso: saber si
> un cliente se ha ido lleva tiempo.
>
> Fíjate en esto —» *(señalar septiembre a diciembre de 2024)* «— el drift sube y sube, cruza el
> umbral en diciembre... y el AUC no se mueve. Cambió la población de clientes, no la relación con
> el churn. Un sistema que reentrenara ante cada alerta de drift habría gastado dinero aquí para
> nada.
>
> Y ahora enero de 2025 —» *(señalar la caída)* «— el AUC se cae seis puntos de golpe. Eso sí es
> concept drift: lo que el modelo aprendió dejó de ser cierto.»

**Si preguntan qué cambió exactamente:** desplegar el detalle del mes. En el baseline,
`support_tickets_30d` era el mejor predictor de comportamiento. En 2025-01 se lanza un programa de
retención proactiva que contacta a quien abre tickets — quejarse deja de ser señal de que te vas y
pasa a ser señal de que te van a salvar. El modelo seguía apoyándose en esa variable.

---

## 0:55 – 1:20 · El modelo en producción

Pestaña **Salud del modelo**.

> «La API no tiene ningún modelo dentro de la imagen. Resuelve `models:/churn-model@champion`
> contra el Model Registry de MLflow al arrancar.
>
> Eso significa que promover un modelo nuevo es mover un alias y llamar a `/reload` —» *(pulsar el
> botón)* «— sin reconstruir la imagen y sin redesplegar. Es lo que hace que el reentrenamiento
> automático no necesite a una persona en medio.»

---

## 1:20 – 1:40 · Predicción en vivo

Pestaña **Predicciones**. Los valores por defecto ya describen a un cliente de riesgo.

> «Contrato mensual, tres meses de antigüedad, cargo alto, tres tickets de soporte.»

Pulsar **Predecir** → 99,7 %.

Cambiar a contrato de dos años, 60 meses de antigüedad, cargo bajo, cero tickets → 0,2 %.

> «Mismo modelo, misma llamada HTTP. El preprocesamiento va dentro del pipeline serializado, así
> que la API manda valores crudos y no puede desincronizarse del entrenamiento.»

---

## 1:40 – 2:00 · Lo que hay detrás

> «Todo esto corre en CI en cada push, incluidas las pruebas de integración con Cloud Storage y
> Secret Manager — contra un emulador de GCP en contenedor, así que no gasto cuota ni expongo
> credenciales en un job que no despliega nada.
>
> Y hay un test que entrena con el periodo baseline y comprueba que el AUC se mantiene durante el
> data drift y cae con el concept drift. Si alguien toca el generador y rompe esa propiedad, se
> entera en CI, no aquí.»

---

## Preguntas que suelen caer

**«¿Por qué gana la regresión logística a XGBoost?»**
Porque el proceso que genera los datos *es* un modelo logístico, así que está correctamente
especificada. Se ve en la métrica `overfit_gap_auc`: −0,005 en la logística frente a +0,096 en
XGBoost. Los modelos de boosting tienen capacidad de sobra y memorizan ruido. Con datos reales el
resultado probablemente sería el contrario, y por eso el pipeline compara los tres en vez de
asumirlo.

**«¿Por qué no reentrenas automáticamente al detectar drift?»**
Porque el propio dataset demuestra que sería un error: hubo cuatro meses de drift de datos con el
modelo perfectamente sano. La política implementada exige una caída de rendimiento para escalar a
crítico; el drift de datos por sí solo nunca pasa de «vigilar».

**«El AUC mensual no lo tendrías en producción.»**
Correcto, y está documentado como tal. Por eso el sistema mide además el drift de predicciones,
que sí está disponible sin etiquetas. En este dataset se dispara en 2024-10, antes de que el
rendimiento caiga.

**«¿Cómo pasarías esto a GCP real?»**
Cambiando variables de entorno. Los SDK de Google leen `STORAGE_EMULATOR_HOST`; si no está,
van a la nube. El workflow de despliegue está escrito en `.github/workflows/deploy.yml`, con
Workload Identity Federation en vez de claves de larga duración. Lo que falta es la cuenta.

**«¿Y si llega una categoría que no existía al entrenar?»**
El `OneHotEncoder` lleva `handle_unknown="ignore"`, así que la codifica como todo ceros y sigue
sirviendo. Hay un test que lo comprueba con una región inventada. Rechazarla obligaría a
redesplegar cada vez que la empresa abre mercado.

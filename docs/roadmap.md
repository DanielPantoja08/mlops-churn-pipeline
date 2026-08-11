# Roadmap: Pipeline MLOps End-to-End — Predicción de Churn

**Stack:** Python · Airflow · MLflow · DVC · FastAPI · Docker · GCP (Cloud Run, Cloud Scheduler, Artifact Registry) · Floci (emulador GCP en contenedor, para CI) · GitHub Actions · Evidently AI · Streamlit

**Objetivo del proyecto:** demostrar dominio del ciclo de vida completo de un modelo de ML en producción — no solo entrenar un modelo, sino explorarlo con criterio, versionarlo, desplegarlo, monitorearlo y reentrenarlo automáticamente cuando el rendimiento se degrada.

---

## Fase 0 — Setup del repositorio

**Objetivo:** estructura profesional desde el día 1.

```
mlops-churn-pipeline/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda/
├── src/
│   ├── data/           # generación e ingesta
│   ├── features/        # feature engineering
│   ├── training/         # entrenamiento
│   └── api/              # FastAPI
├── airflow/
│   └── dags/
├── tests/
├── monitoring/
├── dashboard/
├── .github/workflows/
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── requirements.txt
└── README.md
```

**Checklist:**
- [ ] Crear repo en GitHub (público)
- [ ] `git init`, `.gitignore` (incluir `data/raw/*.csv` si usas DVC)
- [ ] Entorno virtual (`venv` o `poetry`)
- [ ] README inicial con el objetivo del proyecto (lo iremos ampliando)

**Entregable de esta fase:** repo vacío pero con estructura y README con la visión del proyecto.

---

## Fase 1 — Dataset sintético con drift temporal

**Objetivo:** generar 12-18 "meses" de datos de clientes con drift deliberado a partir de cierto mes.

**Pasos:**
1. Definir el esquema de columnas (demográficas, comportamiento, contractuales, `churn`)
2. Escribir `src/data/generate_data.py`:
   - Genera datos base con `numpy`/`faker` para meses 1 al 8 (distribución "normal")
   - A partir del mes 9, inyecta *data drift* (cambia distribución de una variable, ej. uso mensual)
   - A partir del mes 13, inyecta *concept drift* (cambia la relación entre una variable y el churn)
3. Guardar cada mes como `data/raw/2024-XX.csv`
4. Documentar en el script, con comentarios, **qué mes tiene qué tipo de drift y por qué** — esto es tu guion para la demo en la entrevista

**Versionado de datos:**
5. Instalar DVC: `pip install dvc`
6. `dvc init`
7. `dvc add data/raw`
8. Configurar remote de almacenamiento (puede ser un bucket de GCS gratuito)
9. Commit de `.dvc` files a git (no de los CSVs pesados)

**Entregable:** dataset reproducible con `python src/data/generate_data.py`, versionado con DVC.

---

## Fase 2 — EDA (Análisis Exploratorio de Datos) visible

**Objetivo:** mostrar tu proceso de pensamiento antes de modelar, con visualizaciones claras y una narrativa — no una lista de gráficos sin conclusiones. Esta fase es la que muchos portafolios omiten o esconden en un notebook desordenado; aquí la vamos a tratar como un entregable de primera clase.

**Qué significa "visible" en este proyecto:**
- El notebook se renderiza limpio en GitHub (usa Markdown cells entre cada gráfico explicando qué se ve y qué decisión se toma a partir de eso)
- Se exporta una versión HTML navegable, enlazada directamente desde el README principal
- Los 3-4 hallazgos más importantes se resumen en una sección del README con las imágenes embebidas (así el reclutador no necesita abrir el notebook para ver lo esencial)

**Pasos:**
1. `notebooks/eda/01_exploracion_inicial.ipynb`:
   - Forma del dataset, tipos de datos, nulos, duplicados
   - Distribución de la variable objetivo (`churn`) — documentar el desbalance de clases, porque va a justificar decisiones de la Fase 3 (métricas, sampling)
2. `notebooks/eda/02_analisis_univariado.ipynb`:
   - Distribución de cada variable numérica (histogramas, boxplots para detectar outliers)
   - Frecuencias de variables categóricas
3. `notebooks/eda/03_analisis_bivariado.ipynb`:
   - Relación de cada variable con `churn` (ej. tasa de churn por tipo de contrato, por antigüedad)
   - Matriz de correlación entre variables numéricas
   - Al menos un test estadístico (chi-cuadrado para categóricas vs. churn, o punto-biserial para numéricas vs. churn) — esto separa un portafolio "junior" de uno con rigor estadístico
4. `notebooks/eda/04_analisis_temporal.ipynb` **(el más importante para este proyecto en particular)**:
   - Visualiza cómo cambian las distribuciones mes a mes — esta es la prueba visual de que el drift que inyectaste en la Fase 1 existe y es detectable a simple vista, antes incluso de correr Evidently en la Fase 7
   - Gráfico de tasa de churn por mes a lo largo del tiempo
   - Esto conecta directamente el EDA con el problema de negocio que el pipeline entero está resolviendo
5. Cerrar con una sección "Conclusiones y decisiones de feature engineering": lista corta de qué vas a hacer distinto en la Fase 3 por lo que encontraste aquí (ej. "la variable X tiene outliers extremos → aplicar winsorización", "las clases están desbalanceadas 80/20 → usar class_weight o SMOTE")

**Cómo hacerlo visible desde afuera del notebook:**
6. Exportar a HTML: `jupyter nbconvert --to html notebooks/eda/04_analisis_temporal.ipynb`
7. Publicar los HTML exportados con GitHub Pages (gratis, y te da una URL pública tipo `tuusuario.github.io/proyecto/eda`)
8. En el README principal, sección "Exploratory Data Analysis" con:
   - 2-3 gráficos clave embebidos como imágenes (no solo enlaces)
   - Link a la versión completa en GitHub Pages
   - 3-4 bullets con los hallazgos más importantes en texto plano, para quien solo escanea el README

**Entregable:** carpeta de notebooks de EDA limpios y comentados, versión HTML publicada, y resumen visual en el README principal.

---

## Fase 3 — Feature engineering y entrenamiento con tracking

**Objetivo:** pipeline de entrenamiento reproducible con experimentos trackeados, informado directamente por los hallazgos de la Fase 2.

**Pasos:**
1. `src/features/build_features.py`: transformaciones (encoding, escalado, manejo de nulos y outliers) — implementa aquí las decisiones que documentaste al cierre del EDA
2. Instalar MLflow: `pip install mlflow`
3. Levantar MLflow tracking server local: `mlflow ui`
4. `src/training/train.py`:
   - Entrena 3 modelos: regresión logística (baseline), XGBoost, LightGBM
   - Usa `mlflow.log_param()`, `mlflow.log_metric()`, `mlflow.log_model()` en cada run
   - Métricas clave para churn: AUC-ROC, precision, recall, F1 (recuerda el desbalance de clases que viste en el EDA)
5. Comparar runs en la UI de MLflow, elegir el mejor modelo
6. Registrar el modelo ganador en el **Model Registry** de MLflow (`mlflow.register_model()`)

**Entregable:** al menos 3 experimentos trackeados y comparables en MLflow UI, un modelo registrado con versión.

---

## Fase 4 — API de inferencia con FastAPI

**Objetivo:** exponer el modelo como servicio.

**Pasos:**
1. `src/api/main.py` con endpoints:
   - `POST /predict` — recibe features de un cliente, devuelve probabilidad de churn
   - `GET /health` — healthcheck
   - `GET /model-info` — versión del modelo activo, fecha de entrenamiento, métricas
2. Cargar el modelo desde el MLflow Model Registry (no desde un `.pkl` local — esto es clave para que el reentrenamiento automático de la Fase 8 funcione sin redeploy manual)
3. Validación de inputs con Pydantic (schemas claros, con ejemplos)
4. `tests/test_api.py` con pytest: al menos 5 tests (input válido, input inválido, healthcheck, etc.)

**Entregable:** API corriendo localmente (`uvicorn src.api.main:app --reload`), tests pasando.

---

## Fase 5 — Contenerización

**Objetivo:** empaquetar la API para que corra igual en cualquier entorno.

**Pasos:**
1. `Dockerfile` (multi-stage build: una etapa para instalar dependencias, otra más liviana para runtime)
2. `docker-compose.yml` para levantar API + MLflow server juntos localmente
3. Probar: `docker build -t churn-api .` → `docker run -p 8000:8000 churn-api`
4. Verificar que `/predict` responde igual que en local sin Docker

**Entregable:** imagen Docker funcional, documentada en el README (comandos de build y run).

---

## Fase 6 — CI/CD con GitHub Actions

**Objetivo:** automatizar tests, build y push de la imagen en cada cambio.

**Pasos:**
1. `.github/workflows/ci.yml`:
   - Trigger: push a `main` o pull request
   - Job 1: correr `pytest`
   - Job 2: tests de integración contra GCP emulado con **Floci** (ver subsección abajo)
   - Job 3 (solo si Jobs 1 y 2 pasan): build de la imagen Docker
   - Job 4: push a **Artifact Registry** de GCP
2. Configurar secrets en GitHub (credenciales de GCP reales, solo para el job de deploy) — nunca hardcodear
3. Badge de status del build en el README (se ve bien y es funcional)

### Tests de integración con Floci (GCP emulado)

**Objetivo:** validar que la integración con servicios GCP (Cloud Storage, Secret Manager, etc.) funciona *antes* de gastar tiempo o cuota real desplegando contra la nube de verdad.

**Punto clave: Floci corre como contenedor, no como binario instalado en el runner.** En CI se levanta como un servicio adicional del job (o vía `docker run`/`docker-compose`), igual que harías con una base de datos de prueba — nunca se instala directamente en la máquina host.

**Pasos:**
1. En `docker-compose.yml` (para desarrollo local), agregar el servicio:
   ```yaml
   floci-gcp:
     image: floci/floci-gcp:latest
     ports:
       - "4588:4588"
   ```
2. En el workflow de GitHub Actions, declarar Floci como **service container** del job de integración:
   ```yaml
   jobs:
     integration-tests:
       runs-on: ubuntu-latest
       services:
         floci-gcp:
           image: floci/floci-gcp:latest
           ports:
             - 4588:4588
       steps:
         - uses: actions/checkout@v4
         - name: Run integration tests against emulated GCP
           run: pytest tests/integration/ --gcp-endpoint=http://localhost:4588
   ```
3. `tests/integration/test_gcs_upload.py` y similares: apuntan el SDK de GCP (`google-cloud-storage`, etc.) al endpoint de Floci en lugar del real, usando las mismas credenciales dummy (Floci no requiere autenticación)
4. Verificar operaciones críticas del pipeline contra el emulador: subida de artefactos de modelo a Cloud Storage, lectura de secretos, etc.
5. Solo si estos tests pasan, el workflow avanza al job de build/push real

**Por qué vale la pena mencionarlo en el README:** corres pruebas de integración cloud en cada push sin gastar cuota ni depender de que la cuenta de GCP esté disponible — y sin exponer credenciales reales en un job que ni siquiera necesita desplegar nada. Es una práctica de ingeniería que se ve poco en portafolios de ciencia de datos.

**Entregable:** workflow verde en GitHub Actions con un job dedicado de integración contra GCP emulado (Floci en contenedor), y el job de build/deploy separado y dependiente de que esas pruebas pasen.

---

## Fase 7 — Despliegue en GCP

**Objetivo:** API accesible públicamente.

**Pasos:**
1. Crear proyecto en GCP (capa gratuita)
2. Habilitar Cloud Run y Artifact Registry
3. Desplegar: `gcloud run deploy churn-api --image <ruta-de-la-imagen> --region us-central1 --allow-unauthenticated`
4. Configurar variables de entorno (URL del MLflow registry, etc.)
5. Probar el endpoint público con `curl` o Postman
6. (Opcional pero recomendado) Añadir el paso de deploy al workflow de GitHub Actions, para que sea 100% automático tras cada merge a `main`

**Entregable:** URL pública funcional, documentada en el README con ejemplo de request/response.

---

## Fase 8 — Monitoreo de drift con Evidently AI

**Objetivo:** detectar automáticamente cuándo el modelo se degrada — confirmando con métricas lo que ya viste visualmente en la Fase 2.

**Pasos:**
1. `pip install evidently`
2. `monitoring/generate_report.py`:
   - Compara distribución de datos de referencia (mes de entrenamiento) vs. datos "nuevos" (meses simulados con drift)
   - Genera reporte HTML de Evidently con métricas de data drift y prediction drift
3. Definir un umbral de drift (ej. si más del 30% de las columnas muestran drift significativo → alerta)
4. Guardar el resultado del reporte como JSON/métrica consultable (no solo HTML bonito)

**Entregable:** reportes de Evidently generados para al menos 2-3 "meses" del dataset, mostrando cómo el drift aumenta con el tiempo.

---

## Fase 9 — Orquestación con Airflow

**Objetivo:** automatizar el ciclo completo: ingesta → chequeo de drift → reentrenamiento condicional.

**Pasos:**
1. Levantar Airflow (Docker Compose oficial de Airflow es lo más rápido)
2. `airflow/dags/churn_pipeline_dag.py` con tareas:
   - `check_new_data` → ¿hay datos del mes nuevo?
   - `run_drift_report` → corre Evidently
   - `evaluate_drift_threshold` → branch: si supera umbral, continúa; si no, termina
   - `retrain_model` → reentrena con datos actualizados, logea en MLflow
   - `compare_models` → compara el nuevo modelo vs. el actual en producción (por AUC en un set de validación)
   - `promote_model` → si el nuevo modelo es mejor, lo promueve en el MLflow Registry (esto automáticamente actualiza lo que sirve la API, sin redeploy)
3. Programar el DAG (diario o semanal, según cómo hayas simulado los "meses")

**Entregable:** DAG visible y ejecutable en la UI de Airflow, con al menos una corrida completa documentada (screenshot o video corto).

---

## Fase 10 — Dashboard con Streamlit

**Objetivo:** visualización que cierra el loop, pensada para *demo en entrevista*.

**Pasos:**
1. `dashboard/app.py` con 4 secciones:
   - **EDA resumido**: 2-3 gráficos clave de la Fase 2, para que el dashboard cuente la historia completa sin necesitar el notebook
   - **Predicciones recientes**: tabla + distribución de probabilidades de churn
   - **Salud del modelo**: métricas actuales vs. históricas, versión activa
   - **Drift**: gráficos de Evidently embebidos, con indicador visual (verde/amarillo/rojo) del estado del sistema
2. Desplegar el dashboard (puede ir en Cloud Run también, o Streamlit Community Cloud si quieres algo gratis y rápido)

**Entregable:** dashboard público, con datos reales de tu pipeline (no mockeados).

---

## Fase 11 — Documentación final y presentación

**Objetivo:** que cualquier reclutador entienda el proyecto en 2 minutos.

**Pasos:**
1. README principal con:
   - Diagrama de arquitectura (puedes usar draw.io o excalidraw)
   - Sección de EDA con gráficos embebidos y link al análisis completo (ver Fase 2)
   - GIF o video corto (1-2 min) mostrando: request a la API → drift detectado → reentrenamiento → dashboard actualizado
   - Sección "decisiones de diseño" explicando *por qué* elegiste cada herramienta (esto demuestra criterio, no solo ejecución)
   - Instrucciones claras de cómo correrlo localmente
2. Post en LinkedIn explicando el proyecto (opcional pero muy efectivo para visibilidad)

**Entregable final:** repositorio completo, público, con demo grabada.

---

## Orden de prioridad si el tiempo se acaba

Si no llegas a completar todo, este es el orden de valor por esfuerzo, de mayor a menor impacto en entrevistas:

1. Fases 1-2 (dataset con drift + EDA visible) — es la base narrativa de todo el proyecto y lo que primero ve un reclutador
2. Fases 3-5 (entrenamiento trackeado + API + Docker) — con esto ya tienes un proyecto sólido y completo
3. Fase 8 (monitoreo con Evidently) — es lo más "no común" y llamativo, y conecta directamente con el EDA temporal
4. Fase 7 (despliegue en GCP) — tener una URL pública es oro para el CV
5. Fase 6 (CI/CD)
6. Fases 9-10 (Airflow + dashboard) — son el "wow factor" pero también las más pesadas de tiempo

---

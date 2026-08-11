# syntax=docker/dockerfile:1.7

# =============================================================================
# Build multi-etapa
#
# La etapa `builder` trae uv y compila el entorno virtual. La etapa final parte
# de una imagen limpia de Python y solo copia el venv ya resuelto: ni uv, ni
# compiladores, ni caché de paquetes llegan a la imagen que se despliega.
#
# El orden de las capas está pensado para la caché: las dependencias se
# instalan ANTES de copiar el código, así que cambiar un .py no reinstala
# nada — que es lo que pasa el 95 % de las veces.
# =============================================================================

ARG PYTHON_VERSION=3.12

# --- Etapa 1: dependencias ---------------------------------------------------
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder

# Qué extras y grupos instalar. La imagen de la API y la del dashboard salen del
# mismo Dockerfile cambiando solo este argumento.
ARG UV_GROUPS="--extra api"

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Primero solo el manifiesto y el lockfile: esta capa se reutiliza mientras las
# dependencias no cambien. `--frozen` obliga a respetar uv.lock exactamente, sin
# resolver nada por su cuenta: la imagen es reproducible.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --frozen --no-install-project --no-dev ${UV_GROUPS}

# Ahora el código y la instalación del propio proyecto.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable ${UV_GROUPS}

# --- Etapa 2: runtime --------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# libgomp1 lo necesitan LightGBM y XGBoost en tiempo de ejecución. Es la
# dependencia de sistema que más builds de imágenes de ML rompe, porque falla al
# importar y no al instalar.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Usuario sin privilegios. Un proceso que no necesita root no debe correr como
# root, y menos uno expuesto a internet.
RUN useradd --create-home --uid 10001 appuser

# Directorio del backend de MLflow cuando esta misma imagen corre como tracking
# server. Se crea aquí y no en compose porque Docker inicializa el volumen con
# armazón y permisos del directorio de la imagen: si no existiera, el volumen
# nacería propiedad de root y el proceso, que corre sin privilegios, no podría
# escribir la base de datos.
RUN mkdir -p /mlflow && chown appuser:appuser /mlflow

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser examples/ ./examples/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_PORT=8000

USER appuser
EXPOSE 8000

# El healthcheck consulta /health, que devuelve 503 si el modelo no cargó. Así
# Docker distingue "el proceso vive" de "el servicio funciona", que no es lo
# mismo: un proceso arrancado sin modelo está vivo y no sirve para nada.
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=4 \
    CMD curl --fail --silent http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

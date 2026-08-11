# syntax=docker/dockerfile:1.7
#
# Imagen del dashboard. Reutiliza el Dockerfile principal cambiando el grupo de
# dependencias que instala: el dashboard necesita streamlit y plotly, que no
# tienen por qué estar en la imagen de la API.
#
# Se mantiene separado en vez de meter todo en una sola imagen porque la API es
# lo que se despliega y se escala: cargarla con las dependencias de un panel que
# usan tres personas sería pagar arranques más lentos en cada instancia.

ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --frozen --no-install-project --no-dev --extra api --group dashboard

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra api --group dashboard

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser dashboard/ ./dashboard/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=4 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

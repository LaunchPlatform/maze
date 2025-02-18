FROM python:3.11.11-alpine3.21

ENV PYTHONPATH=/app
ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN apk update && apk add --no-cache \
        uv \
        gcc \
        ca-certificates \
        curl \
        bash \
        musl-dev \
        python3-dev \
        libffi-dev

WORKDIR /app

# ref: https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

COPY . /app/

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

CMD ["uvicorn", "--factory", "maze.web.app:make_app", "--port=8080", "--host=0.0.0.0"]

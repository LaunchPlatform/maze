FROM python:3.11.11-bookworm

ENV PYTHONPATH=/app
ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN pip install uv

# ref: https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --extra torch-gpu \

COPY . /app/
EXPOSE 8080

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra torch-gpu

# FIXME:
CMD ["uvicorn", "--factory", "maze.web.app:make_app", "--port=8080", "--host=0.0.0.0"]

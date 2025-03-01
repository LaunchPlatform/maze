FROM python:3.11.11-bookworm

ENV PYTHONPATH=/app

RUN pip install uv
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --system

WORKDIR /app
EXPOSE 8080

COPY . /app/

RUN uv pip compile pyproject.toml -o requirements.txt && \
  uv pip install -r requirements.txt --system
 
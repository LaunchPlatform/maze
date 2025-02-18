FROM python:3.11.9-alpine3.19

ENV PYTHONPATH=/app

RUN apk update && apk add --no-cache \
        gcc \
        ca-certificates \
        curl \
        bash \
        musl-dev \
        python3-dev \
        libffi-dev

COPY . /app/
WORKDIR /app
RUN pip install poetry
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

CMD ["uvicorn", "--factory", "maze.web.app:make_app", "--port=8080", "--host=0.0.0.0"]

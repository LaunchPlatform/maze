FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV PYTHONPATH=/app

COPY . /app/
WORKDIR /app
RUN pip install poetry
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi


FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

COPY . /app/
WORKDIR /app
RUN pip install poetry
RUN poetry install --no-interaction --no-ansi

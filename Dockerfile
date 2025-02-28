FROM python:3.11.11-bookworm

ENV PYTHONPATH=/app

RUN pip install uv

WORKDIR /app
EXPOSE 8080

COPY . /app/

RUN uv pip compile pyproject.toml -o requirements.txt --extra torch-gpu

# FIXME:
CMD ["uvicorn", "--factory", "maze.web.app:make_app", "--port=8080", "--host=0.0.0.0"]

FROM 022497628448.dkr.ecr.us-west-2.amazonaws.com/pet-projects/maze-base:0.0.1

ENV PYTHONPATH=/app
WORKDIR /app
EXPOSE 8080

COPY . /app/

RUN uv pip compile pyproject.toml -o requirements.txt && \
  uv pip install -r requirements.txt --system

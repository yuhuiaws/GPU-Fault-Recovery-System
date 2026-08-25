FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir '.[collectors,postgres]'

ENTRYPOINT ["gpu-fault-collector"]

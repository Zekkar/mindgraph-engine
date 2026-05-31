FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install -e . --no-cache-dir
COPY mindgraph/ ./mindgraph/
CMD ["mindgraph", "serve", "--config", "/data/mindgraph.yml", "--base-dir", "/data"]

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install -e . --no-cache-dir
COPY mindgraph/ ./mindgraph/
# 0.0.0.0 is required INSIDE the container for the port mapping to work; the
# compose file binds the published port to host loopback so it is not LAN-exposed.
CMD ["mindgraph", "serve", "--host", "0.0.0.0", "--config", "/data/mindgraph.yml", "--base-dir", "/data"]

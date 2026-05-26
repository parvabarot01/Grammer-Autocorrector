FROM python:3.10-slim AS base

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS api

COPY src/ ./src/
COPY data/sample/ ./data/sample/
COPY data/prompt_registry.json ./data/prompt_registry.json

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

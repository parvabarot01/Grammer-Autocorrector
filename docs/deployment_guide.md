# Deployment Guide

## Overview
This guide covers local development, Docker deployment, environment variables, health checks, and scaling considerations for the Grammar Autocorrector API and Gradio UI.

## Local Development

### Prerequisites

- Python `3.10+`
- `pip`
- Optional: virtual environment tool such as `venv`

### Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run the UI

```bash
python src/ui/gradio_app.py
```

## Docker Deployment

### Dockerfile
The project root includes a multi-stage `Dockerfile` that installs dependencies once and runs either the API or the UI from the same image.

### Build the API image

```bash
docker build --target api -t grammar-autocorrector-api .
```

### Run the API container

```bash
docker run --rm -p 8000:8000 grammar-autocorrector-api
```

### Docker Compose
The project root includes `docker-compose.yml` with two services:

- `api`: FastAPI backend
- `ui`: Gradio frontend configured to call `http://api:8000`

Start both services:

```bash
docker-compose up --build
```

Service URLs:

- API docs: `http://localhost:8000/docs`
- UI: `http://localhost:7860`

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `T5_MODEL_NAME` | `t5-base` | T5 correction model id or local path |
| `BERT_MODEL_NAME` | `bert-base-uncased` | BERT detector model id or local path |
| `MODEL_MAX_LENGTH` | `128` | Shared tokenizer length limit |
| `MODEL_NUM_BEAMS` | `4` | Default T5 beam width |
| `MODEL_LEARNING_RATE` | `3e-4` | Training learning rate |
| `MODEL_BATCH_SIZE` | `16` | Default batch size |
| `MODEL_EPOCHS` | `5` | Default training epochs |
| `RAW_DATA_PATH` | `data/raw` | Raw dataset directory |
| `PROCESSED_DATA_PATH` | `data/processed` | Processed dataset directory |
| `SAMPLE_DATA_PATH` | `data/sample` | Sample assets and rule files |
| `RAG_EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model id |
| `RAG_VECTOR_STORE_PATH` | `data/vector_store` | RAG index storage directory |
| `RAG_TOP_K` | `5` | Default retrieval depth |
| `RAG_CHUNK_SIZE` | `512` | Character chunk size |
| `RAG_CHUNK_OVERLAP` | `50` | Chunk overlap |
| `API_HOST` | `0.0.0.0` | API host binding |
| `API_PORT` | `8000` | API port |
| `API_MAX_CONCURRENT_REQUESTS` | `100` | Planned concurrency guard |
| `GUARDRAILS_TOXICITY_THRESHOLD` | `0.8` | Toxicity block threshold |
| `GUARDRAILS_MAX_INPUT_LENGTH` | `1000` | Maximum request input length |
| `API_URL` | `http://localhost:8000` | UI-side backend base URL |
| `MODEL_DIR` | `/app/models` | Container model mount path |
| `LOG_LEVEL` | `INFO` | Container logging verbosity |

## Health Checks

Use the health endpoint to validate that the container is reachable:

```bash
curl http://localhost:8000/health
```

Expected response shape:

```json
{
  "status": "ok",
  "models_loaded": false,
  "version": "1.0.0",
  "timestamp": "2026-05-25T12:00:00+00:00"
}
```

## Scaling Considerations

- The API layer is stateless apart from mounted models and local vector-store files.
- Horizontal scaling is straightforward if all replicas mount the same model and data volumes.
- For production, place the API behind a reverse proxy or load balancer.
- Rate limiting and authentication are intentionally deferred to v2.
- If FAISS indexes become large, move vector storage to a shared volume or a dedicated retrieval service.

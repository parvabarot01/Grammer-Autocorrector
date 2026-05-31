# Deployment Guide

## Overview
This guide covers local development, Docker deployment, environment variables,
health checks, and scaling considerations for the public Next.js frontend,
FastAPI backend, and optional legacy Gradio UI.

## Local Development

### Prerequisites

- Python `3.10+`
- `pip`
- Node.js `20+`
- `npm`
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

### Run the public frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

### Run the legacy Gradio UI

```bash
python src/ui/gradio_app.py
```

The legacy UI remains useful for operator workflows such as prompt management,
knowledge-base inspection, and evaluation.

## Docker Deployment

### Dockerfile
The project root includes a multi-stage `Dockerfile` for the Python API and
legacy Gradio UI. `frontend/Dockerfile` builds the public Next.js application
with Node.js.

### Build the API image

```bash
docker build --target api -t grammar-autocorrector-api .
```

### Run the API container

```bash
docker run --rm -p 8000:8000 grammar-autocorrector-api
```

### Docker Compose
The project root includes `docker-compose.yml` with three services:

- `api`: FastAPI backend
- `frontend`: public Next.js UI configured to call `http://localhost:8000`
- `ui`: optional legacy Gradio UI configured to call `http://api:8000`

Start both services:

```bash
docker-compose up --build
```

Service URLs:

- Public frontend: `http://localhost:3000`
- API docs: `http://localhost:8000/docs`
- Legacy Gradio UI: `http://localhost:7860`

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
| `ENABLE_PUBLIC_API` | `true` | Enable the public-safe correction route |
| `SHOW_MODEL_DETAILS` | `false` | Reserve internal model details for operator surfaces |
| `FRONTEND_ORIGIN` | `http://localhost:3000` | Browser origin allowed by API CORS |
| `GUARDRAILS_TOXICITY_THRESHOLD` | `0.8` | Toxicity block threshold |
| `GUARDRAILS_MAX_INPUT_LENGTH` | `1000` | Maximum request input length |
| `API_URL` | `http://localhost:8000` | UI-side backend base URL |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Browser-accessible API URL for the Next.js frontend |
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

## Cloud Deployment Status

Cloud deployment is intentionally deferred. Sprint 9 provides local and
containerized full-stack deployment only; a free hosted deployment plan will be
handled in a later sprint.

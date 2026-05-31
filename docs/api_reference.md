# API Reference

## Overview
The Grammar Autocorrector API exposes correction, detection, prompt management,
knowledge-base retrieval, evaluation, and benchmarking over HTTP.

- Base URL: `http://localhost:8000`
- OpenAPI docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Authentication: none in `v1.0.0`
- Rate limiting: planned for `v2` at `100 req/min per IP`
- Versioning strategy: backward-compatible additions remain on `v1`; breaking
  changes will move to `/v2`

## Common Error Codes

| Code | Meaning |
|------|---------|
| `400` | Request payload is valid JSON but invalid for the requested operation |
| `422` | Guardrails blocked the request |
| `500` | Internal pipeline or service error |

## Response Notes

- `POST /public/correct` is the public product contract. Its response is
  intentionally limited to corrected text, user-friendly changes, and a summary.
  It never exposes model, prompt, retrieval, dataset, evaluation, or guardrail
  internals.
- `guardrail_report` contains the full input/output safety assessment returned
  by the unified `CorrectionPipeline`.
- `model_version` identifies the correction path used, such as
  `heuristic-t5-fallback` or a loaded model checkpoint name.
- `prompt_version` is populated for `rag` mode and for the active prompt used
  in `auto` mode when retrieval is selected.
- `request_id` is attached by middleware and mirrored in the response body for
  single-item correction requests.

## POST `/public/correct`

- Purpose: correct text for the public Next.js product UI
- Visibility: public-safe response with internal pipeline details removed
- Request schema:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `text` | `str` | Yes | Trimmed text, `1-1000` chars |

- Response schema:

| Field | Type | Description |
|-------|------|-------------|
| `original_text` | `str` | Sanitized input text |
| `corrected_text` | `str` | Improved text |
| `changes` | `list[PublicCorrectionChange]` | User-friendly before and after fragments |
| `summary` | `str` | Short result summary |
| `success` | `bool` | Whether correction completed |

Each `PublicCorrectionChange` contains `before`, `after`, and `explanation`.

```bash
curl -X POST http://localhost:8000/public/correct \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"She go to school.\"}"
```

```json
{
  "original_text": "She go to school.",
  "corrected_text": "She goes to school.",
  "changes": [
    {
      "before": "go",
      "after": "goes",
      "explanation": "Grammar corrected for clarity and correctness."
    }
  ],
  "summary": "1 grammar issue corrected.",
  "success": true
}
```

## GET `/health`

- Purpose: readiness and model-load status
- Response schema:

| Field | Type | Description |
|-------|------|-------------|
| `status` | `str` | Service state, usually `ok` |
| `models_loaded` | `bool` | Whether pipeline models/resources are loaded |
| `version` | `str` | API version |
| `timestamp` | `str` | ISO 8601 UTC timestamp |

```bash
curl http://localhost:8000/health
```

```python
import requests

print(requests.get("http://localhost:8000/health", timeout=10).json())
```

## GET `/info`

- Purpose: runtime metadata and supported capabilities
- Response schema:

| Field | Type | Description |
|-------|------|-------------|
| `models` | `list[str]` | Loaded component names |
| `prompt_version` | `str` | Active prompt version |
| `capabilities` | `list[str]` | Supported feature flags |

```bash
curl http://localhost:8000/info
```

```python
import requests

print(requests.get("http://localhost:8000/info", timeout=10).json())
```

## POST `/correct`

- Purpose: correct a single text input
- Request schema:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `text` | `str` | Yes | `1-1000` chars |
| `mode` | `"t5" \| "rag" \| "auto"` | No | Default `auto` |
| `num_beams` | `int` | No | Default `4`, range `1-8` |
| `return_detected_errors` | `bool` | No | Default `false` |
| `prompt_version` | `str` | No | Optional prompt override |

- Response schema:

| Field | Type |
|-------|------|
| `original` | `str` |
| `corrected` | `str` |
| `mode_used` | `str` |
| `errors_detected` | `list[ErrorSpan] \| null` |
| `guardrail_report` | `FullGuardrailReport` |
| `processing_time_ms` | `float` |
| `model_version` | `str` |
| `prompt_version` | `str` |
| `request_id` | `str` |

```bash
curl -X POST http://localhost:8000/correct \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"She go to school everyday.\",\"mode\":\"auto\",\"return_detected_errors\":true}"
```

```python
import requests

payload = {
    "text": "She go to school everyday.",
    "mode": "auto",
    "num_beams": 4,
    "return_detected_errors": True,
}
response = requests.post(
    "http://localhost:8000/correct",
    json=payload,
    timeout=30,
)
print(response.json())
```

## POST `/correct/batch`

- Purpose: correct up to `50` inputs in one request
- Request schema:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `texts` | `list[str]` | Yes | max `50` items |
| `mode` | `"t5" \| "rag" \| "auto"` | No | Default `auto` |
| `batch_size` | `int` | No | Default `16` |

- Response schema:

| Field | Type |
|-------|------|
| `results` | `list[BatchCorrectionItem]` |
| `total_processed` | `int` |
| `processing_time_ms` | `float` |

Each `BatchCorrectionItem` includes:

| Field | Type |
|-------|------|
| `original` | `str` |
| `corrected` | `str` |
| `mode_used` | `str` |
| `errors_detected` | `list[ErrorSpan] \| null` |
| `guardrail_report` | `FullGuardrailReport \| null` |
| `processing_time_ms` | `float` |
| `status` | `str` |
| `error_message` | `str \| null` |
| `model_version` | `str` |
| `prompt_version` | `str` |

```bash
curl -X POST http://localhost:8000/correct/batch \
  -H "Content-Type: application/json" \
  -d "{\"texts\":[\"She go home.\",\"He have a apple.\"],\"mode\":\"auto\"}"
```

```python
import requests

response = requests.post(
    "http://localhost:8000/correct/batch",
    json={"texts": ["She go home.", "He have a apple."], "mode": "auto"},
    timeout=30,
)
print(response.json())
```

## POST `/detect`

- Purpose: run token-level grammar error detection without correction
- Request schema:

| Field | Type | Required |
|-------|------|----------|
| `text` | `str` | Yes |

- Response schema:

| Field | Type |
|-------|------|
| `has_errors` | `bool` |
| `errors` | `list[ErrorSpan]` |
| `error_count` | `int` |
| `processing_time_ms` | `float` |

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"She go to school everyday.\"}"
```

```python
import requests

print(
    requests.post(
        "http://localhost:8000/detect",
        json={"text": "She go to school everyday."},
        timeout=30,
    ).json()
)
```

## POST `/knowledge/add`

- Purpose: append grammar rules to the RAG knowledge base
- Request schema:

| Field | Type | Required |
|-------|------|----------|
| `rules` | `list[str]` | Yes |

- Response schema:

| Field | Type |
|-------|------|
| `added` | `int` |
| `total_rules` | `int` |

```bash
curl -X POST http://localhost:8000/knowledge/add \
  -H "Content-Type: application/json" \
  -d "{\"rules\":[\"Use 'an' before a vowel sound.\"]}"
```

```python
import requests

print(
    requests.post(
        "http://localhost:8000/knowledge/add",
        json={"rules": ["Use 'an' before a vowel sound."]},
        timeout=30,
    ).json()
)
```

## GET `/knowledge/search`

- Purpose: retrieve grammar guidance from the RAG knowledge base
- Query parameters:

| Name | Type | Required | Default |
|------|------|----------|---------|
| `query` | `str` | Yes | - |
| `top_k` | `int` | No | `5` |

- Response schema:

| Field | Type |
|-------|------|
| `results` | `list[RetrievedChunk]` |
| `query` | `str` |

```bash
curl "http://localhost:8000/knowledge/search?query=She%20go%20to%20school%20everyday.&top_k=3"
```

```python
import requests

print(
    requests.get(
        "http://localhost:8000/knowledge/search",
        params={"query": "She go to school everyday.", "top_k": 3},
        timeout=30,
    ).json()
)
```

## GET `/prompts`

- Purpose: list all registered prompts and the active version
- Response schema:

| Field | Type |
|-------|------|
| `versions` | `list[PromptVersion]` |
| `active_version` | `str` |

```bash
curl http://localhost:8000/prompts
```

```python
import requests

print(requests.get("http://localhost:8000/prompts", timeout=10).json())
```

## GET `/prompts/{version_id}`

- Purpose: fetch a specific prompt template and metadata
- Path parameter: `version_id`
- Response schema: `PromptVersion`

```bash
curl http://localhost:8000/prompts/v1.1.0
```

```python
import requests

print(requests.get("http://localhost:8000/prompts/v1.1.0", timeout=10).json())
```

## POST `/prompts/{version_id}/promote`

- Purpose: promote a prompt to active production status
- Path parameter: `version_id`
- Response schema:

| Field | Type |
|-------|------|
| `promoted` | `str` |
| `previous` | `str \| null` |

```bash
curl -X POST http://localhost:8000/prompts/v2.0.0/promote
```

```python
import requests

print(
    requests.post(
        "http://localhost:8000/prompts/v2.0.0/promote",
        timeout=10,
    ).json()
)
```

## POST `/prompts/rollback`

- Purpose: roll back to the previously active prompt version
- Response schema:

| Field | Type |
|-------|------|
| `rolled_back_to` | `str` |

```bash
curl -X POST http://localhost:8000/prompts/rollback
```

```python
import requests

print(requests.post("http://localhost:8000/prompts/rollback", timeout=10).json())
```

## POST `/evaluate`

- Purpose: compute offline metrics for predictions against references
- Request schema:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `predictions` | `list[str]` | Yes | Must align with `references` |
| `references` | `list[str]` | Yes | Single reference per prediction |
| `metrics` | `list[str]` | Yes | Supported: `gleu`, `rouge`, `exact_match` |

- Response schema:

| Field | Type |
|-------|------|
| `metrics` | `dict[str, object]` |
| `evaluated_pairs` | `int` |

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d "{\"predictions\":[\"She goes to school every day.\"],\"references\":[\"She goes to school every day.\"],\"metrics\":[\"gleu\",\"rouge\",\"exact_match\"]}"
```

```python
import requests

payload = {
    "predictions": ["She goes to school every day."],
    "references": ["She goes to school every day."],
    "metrics": ["gleu", "rouge", "exact_match"],
}
print(requests.post("http://localhost:8000/evaluate", json=payload, timeout=30).json())
```

## POST `/benchmark`

- Purpose: benchmark the full correction pipeline against labeled pairs
- Request schema:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `test_pairs` | `list[BenchmarkPair]` | Yes | Each item has `original` and `reference` |
| `max_samples` | `int` | No | Optional cap on processed pairs |

- Response schema:

| Field | Type |
|-------|------|
| `gleu` | `float` |
| `rouge` | `dict[str, float]` |
| `exact_match` | `float` |
| `avg_latency_ms` | `float` |
| `p95_latency_ms` | `float` |
| `failure_rate` | `float` |
| `total_samples` | `int` |
| `timestamp` | `str` |

```bash
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d "{\"test_pairs\":[{\"original\":\"She go to school everyday.\",\"reference\":\"She goes to school every day.\"}],\"max_samples\":1}"
```

```python
import requests

payload = {
    "test_pairs": [
        {
            "original": "She go to school everyday.",
            "reference": "She goes to school every day.",
        }
    ],
    "max_samples": 1,
}
print(requests.post("http://localhost:8000/benchmark", json=payload, timeout=60).json())
```

## Data Model Snapshot

### `ErrorSpan`

| Field | Type |
|-------|------|
| `start` | `int` |
| `end` | `int` |
| `token` | `str` |
| `confidence` | `float` |
| `error_type` | `str` |

### `PromptVersion`

| Field | Type |
|-------|------|
| `version_id` | `str` |
| `template` | `str` |
| `description` | `str` |
| `created_at` | `str` |
| `metrics` | `dict[str, float]` |
| `is_active` | `bool` |

### `RetrievedChunk`

| Field | Type |
|-------|------|
| `text` | `str` |
| `score` | `float` |
| `source` | `str` |
| `chunk_id` | `int` |

## Notes for v2

- JWT-based authentication is planned but not required in `v1.0.0`
- Per-IP rate limiting is planned but not yet implemented
- Streaming corrections and async batch jobs are future enhancements

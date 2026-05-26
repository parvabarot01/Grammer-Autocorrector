# API Reference

## Overview
The Grammar Autocorrector API exposes grammar correction, error detection, knowledge-base search, prompt management, and evaluation endpoints over HTTP.

- Base URL: `http://localhost:8000`
- OpenAPI docs: `http://localhost:8000/docs`
- Authentication: None in v1
- Rate limiting: Planned for v2 at `100 req/min per IP`
- Versioning strategy: Pathless semantic versioning for v1 with backward-compatible additions; breaking changes will move to `/v2`

## Common Error Codes

| Code | Meaning |
|------|---------|
| `400` | Request payload is syntactically valid but semantically invalid for the operation |
| `422` | Guardrail validation blocked the request |
| `500` | Internal service or model error |

## GET `/health`

- Purpose: readiness and model-load status
- Response fields:
  - `status`: service status string
  - `models_loaded`: whether T5 and BERT loaded successfully
  - `version`: API version
  - `timestamp`: UTC timestamp

```bash
curl http://localhost:8000/health
```

```python
import requests
print(requests.get("http://localhost:8000/health", timeout=10).json())
```

## GET `/info`

- Purpose: runtime metadata and supported capabilities
- Response fields:
  - `models`
  - `prompt_version`
  - `capabilities`

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
| `prompt_version` | `str` | No | Used in `rag` mode |

- Response schema:

| Field | Type |
|-------|------|
| `original` | `str` |
| `corrected` | `str` |
| `mode_used` | `str` |
| `errors_detected` | `list[ErrorSpan] \| null` |
| `guardrail_report` | `FullGuardrailReport` |
| `processing_time_ms` | `float` |
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
- Query params:

| Name | Type | Required | Default |
|------|------|----------|---------|
| `query` | `str` | Yes | — |
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
- Path params:

| Name | Type | Required |
|------|------|----------|
| `version_id` | `str` | Yes |

```bash
curl http://localhost:8000/prompts/v1.1.0
```

```python
import requests
print(requests.get("http://localhost:8000/prompts/v1.1.0", timeout=10).json())
```

## POST `/prompts/{version_id}/promote`

- Purpose: promote a prompt version to production
- Response schema:

| Field | Type |
|-------|------|
| `promoted` | `str` |
| `previous` | `str` |

```bash
curl -X POST http://localhost:8000/prompts/v2.0.0/promote
```

```python
import requests
print(requests.post("http://localhost:8000/prompts/v2.0.0/promote", timeout=10).json())
```

## POST `/prompts/rollback`

- Purpose: restore the previous active prompt
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

- Purpose: score predictions against references
- Request schema:

| Field | Type | Required |
|-------|------|----------|
| `predictions` | `list[str]` | Yes |
| `references` | `list[str]` | Yes |
| `metrics` | `list["gleu" \| "rouge" \| "exact_match"]` | Yes |

- Response schema:

| Field | Type |
|-------|------|
| `metrics` | `dict` |
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

## Schema Notes

### `ErrorSpan`

| Field | Type | Meaning |
|-------|------|---------|
| `start` | `int` | Start character offset |
| `end` | `int` | End character offset |
| `token` | `str` | Original token text |
| `confidence` | `float` | Error-class confidence |
| `error_type` | `str` | Error label |

### `FullGuardrailReport`

| Field | Type | Meaning |
|-------|------|---------|
| `input_valid` | `GuardrailResult` | Input validation status |
| `toxicity` | `ToxicityResult` | Keyword-based toxicity scan |
| `bias` | `BiasResult` | Simple bias scan |
| `output_valid` | `GuardrailResult \| null` | Output validation status |
| `overall_passed` | `bool` | Aggregate guardrail result |
| `timestamp` | `str` | UTC timestamp |

## Programmatic Utilities
The project also exposes Python utility APIs used by training and evaluation workflows:

- `src.utils.preprocessing.GrammarPreprocessor`
- `src.utils.evaluation.Evaluator`
- `src.utils.config.load_config`

Those modules remain documented with docstrings and type hints in source.

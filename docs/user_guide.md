# User Guide

## Getting Started

1. Install dependencies with `pip install -r requirements.txt`.
2. Start the API with `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`.
3. Run `cd frontend`, then install UI dependencies with `npm install`.
4. Start the public UI with `npm run dev`.
5. Open `http://localhost:3000` for the product UI or `http://localhost:8000/docs` for the API docs.

If trained weights are not available locally, the system still runs using lightweight correction fallbacks so you can explore the end-to-end experience.

## First Correction

Public single-request example:

```bash
curl -X POST http://localhost:8000/public/correct \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"She go to school everyday.\"}"
```

The public response includes the original text, corrected text, a simple list of
changes, and a short summary. Internal model and pipeline details are
intentionally hidden from public users.

## Public Next.js UI

1. Open `http://localhost:3000`.
2. Select `Try correction` or open `Correct Text`.
3. Enter a sentence or short paragraph.
4. Select `Correct text`.
5. Review the improved text and the highlighted before/after changes.
6. Select `Copy corrected text` to use the polished result elsewhere.

The public UI includes loading and error states, a character counter, a reset
button, and a responsive layout for mobile and desktop screens.

## Legacy Gradio Operator UI

Start the optional operator UI with:

```bash
python src/ui/gradio_app.py
```

Open `http://localhost:7860`. This interface remains available for internal
testing and operational workflows.

### Correct Text
Screenshot description: a multiline text box at the top, mode and decoding controls beneath it, corrected text on the right, highlighted error spans, and a JSON guardrail panel below.

- Use this tab for one-off correction requests.
- Select `Auto`, `T5 Only`, or `RAG-Enhanced`.
- Turn on error highlighting if you want token-level feedback.

### Batch Correct
Screenshot description: a spreadsheet-style table for multiple inputs on the left and a results table with status labels on the right.

- Paste up to 50 input texts.
- Run batch processing and download a CSV of the results.
- Failed items are returned with item-level error metadata instead of breaking the whole batch.

### Detect Errors
Screenshot description: one text box, a highlighted output view, and a JSON list of detected spans and confidence values.

- Use this when you only need detection, not correction.
- Good for debugging labeled examples before training.

### Knowledge Base
Screenshot description: a rule-entry text area, a search query box, and JSON panels showing add/search results.

- Add new grammar rules line by line.
- Search for relevant rules that the RAG pipeline would retrieve during correction.

### Prompt Manager
Screenshot description: a prompt-version dropdown, current template viewer, active-version field, action buttons, and a version-history grid.

- Review prompt templates.
- Promote a new prompt to production.
- Roll back if a new prompt underperforms.

### Evaluate
Screenshot description: a file uploader for prediction/reference CSV files, metric checkboxes, a bar chart, and a downloadable report file.

- Upload a CSV containing `prediction` and `reference` columns.
- Choose `gleu`, `rouge`, and/or `exact_match`.
- Download the generated evaluation output.

## REST API

### POST `/public/correct`
- Corrects text for the public product UI.
- Returns only original text, corrected text, simple changes, a summary, and success status.
- Does not expose model names, prompt versions, retrieval details, metrics, datasets, or guardrail internals.

### GET `/health`
- Use it for health checks and readiness probes.
- Returns service status, version, timestamp, and whether primary models loaded.

### GET `/info`
- Returns active prompt version and supported capabilities.

### POST `/correct`
- Corrects one input string for internal and operator workflows.
- Supports `auto`, `t5`, and `rag` modes.

### POST `/correct/batch`
- Corrects up to 50 inputs in one request.
- Returns per-item results with success or error status.

### POST `/detect`
- Returns token-level error spans without rewriting the text.

### POST `/knowledge/add`
- Adds grammar rules to the retrieval knowledge base.

### GET `/knowledge/search`
- Retrieves the top matching grammar rules for a query.

### GET `/prompts`
- Lists all prompt versions and the active production prompt.

### GET `/prompts/{version_id}`
- Returns one prompt template and its metadata.

### POST `/prompts/{version_id}/promote`
- Activates a specific prompt version.

### POST `/prompts/rollback`
- Restores the previous active prompt.

### POST `/evaluate`
- Computes evaluation metrics over prediction/reference pairs.

### POST `/benchmark`
- Runs correction over benchmark pairs and returns quality and latency stats.

## Understanding `CorrectionResult`

| Field | Meaning |
|-------|---------|
| `original` | Sanitized version of the input text |
| `corrected` | Final corrected text returned to the caller |
| `mode_used` | Actual correction path used |
| `errors_detected` | Optional token-level error spans |
| `guardrail_report` | Input/output safety and validation report |
| `processing_time_ms` | Total request latency |
| `model_version` | T5 model id or checkpoint reference |
| `prompt_version` | Active or explicitly selected prompt version |

## Prompt Versioning Workflow

1. Register or edit a prompt template in the prompt registry.
2. Evaluate it on a validation set or benchmark endpoint.
3. Update stored metrics if needed.
4. Promote the candidate version.
5. Roll back if quality regresses.

## Knowledge Base Management

- Store reusable grammar rules in concise, one-rule-per-line format.
- Prefer domain-specific rules for specialized writing styles.
- Re-index after major rule updates by reloading the pipeline or restarting the API.
- Use `/knowledge/search` to confirm retrieval quality before pushing prompt changes.

## Troubleshooting

1. `422` on correction requests usually means a guardrail blocked the input.
2. Empty correction output usually indicates an upstream validation or output-guardrail failure.
3. `models_loaded=false` on `/health` means trained weights are not installed or not reachable.
4. Slow first request is expected because the pipeline may initialize retrieval assets.
5. If the RAG path returns generic output, verify that `data/sample/grammar_rules.txt` exists or add domain rules manually.
6. If `/benchmark` returns weak scores, confirm that trained checkpoints are available and not just heuristic fallbacks.
7. If prompt rollback seems to do nothing, check whether the history contains more than one active version.
8. If CSV evaluation fails, verify the file has `prediction` and `reference` columns.
9. If Docker health checks fail, confirm the API container can bind to port `8000`.
10. If Jupyter or notebook tooling behaves strangely after dependency installs, use a clean virtual environment for this project.

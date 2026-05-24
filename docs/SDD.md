# Software Design Document

## 1. System Architecture

The Grammar Autocorrector System follows a modular architecture separating model inference, retrieval and prompt management, serving interfaces, and supporting utilities. The design supports both research iteration and operational deployment by keeping data, models, pipeline logic, APIs, and UI components in dedicated layers.

### High-Level Architecture

```text
Client/UI
   |
   v
FastAPI Layer ---------------------------+
   |                                     |
   v                                     v
Correction Services                Evaluation Services
   |                                     |
   +-----------+--------------+----------+
               |              |
               v              v
        BERT Detector     T5 Corrector
               |              |
               +-------> RAG Pipeline <------ Prompt Versioning
                              |
                              v
                         Guardrails
                              |
                              v
                      Response Serialization
```

## 2. Component Design

### 2.1 T5 Corrector Module

Purpose:
Generate corrected text from grammatically incorrect input using a fine-tuned T5 sequence-to-sequence model.

Responsibilities:

- Load tokenizer and model weights.
- Prefix inputs for task-specific prompting.
- Perform beam-search inference.
- Fine-tune on benchmark datasets and track metrics.

Inputs:
Raw text or batched text samples.

Outputs:
Corrected text strings and training/evaluation metadata.

### 2.2 BERT Detector Module

Purpose:
Classify tokens as correct or erroneous to provide lightweight error localization and support auto-mode routing.

Responsibilities:

- Perform token-level inference.
- Return spans, token indices, and confidence scores.
- Support fine-tuning with class imbalance handling.

Inputs:
Raw text strings.

Outputs:
Structured error spans and boolean error signals.

### 2.3 RNN Baseline Module

Purpose:
Provide a simpler encoder-decoder baseline for benchmarking and teaching value.

Responsibilities:

- Build vocabulary and embeddings.
- Run sequence-to-sequence training with attention.
- Produce baseline correction outputs.

Inputs:
Tokenized or raw training text pairs.

Outputs:
Baseline corrected sequences and loss metrics.

### 2.4 RAG Pipeline Module

Purpose:
Augment correction prompts with retrieved grammar rules or contextual examples.

Responsibilities:

- Chunk knowledge documents.
- Create and persist embeddings.
- Search a FAISS vector index.
- Inject retrieved context into prompt templates.

Inputs:
Knowledge documents and user query text.

Outputs:
Retrieved chunks, augmented prompts, and RAG-assisted corrections.

### 2.5 Prompt Versioning Module

Purpose:
Track prompt templates as versioned assets with metrics and promotion state.

Responsibilities:

- Register prompt versions.
- Mark active production prompt.
- Roll back to earlier versions.
- Export registry state.

Inputs:
Prompt templates, metadata, evaluation scores.

Outputs:
Version registry entries and active prompt selection.

### 2.6 Guardrails Module

Purpose:
Apply responsible AI checks on input and output text.

Responsibilities:

- Validate size, character set, and prompt-injection patterns.
- Score basic toxicity and bias heuristics.
- Sanitize content where safe and appropriate.

Inputs:
Original user text and corrected output.

Outputs:
Structured validation and policy reports.

### 2.7 FastAPI Layer

Purpose:
Expose application capabilities through REST endpoints and OpenAPI documentation.

Responsibilities:

- Validate requests and serialize responses.
- Route requests to the correction pipeline.
- Surface errors and timing metadata.

Inputs:
HTTP requests from clients.

Outputs:
JSON responses, health signals, and operational metadata.

### 2.8 Gradio UI Layer

Purpose:
Provide a fast, interactive frontend for demos, debugging, and stakeholder review.

Responsibilities:

- Render correction and batch workflows.
- Display error highlights and guardrail reports.
- Trigger API calls for core operations.

Inputs:
Text entered by users or uploaded batch files.

Outputs:
Corrected text, summaries, tables, and downloadable artifacts.

## 3. Data Flow

```text
Input Text
   |
   v
Preprocessing and Input Validation
   |
   v
BERT Error Detection
   |
   +--> no errors in auto mode --> original text returned
   |
   v
Correction Selection
   |
   +--> T5 direct correction
   |
   +--> RNN benchmark correction
   |
   +--> RAG retrieval -> prompt assembly -> context-aware correction
   |
   v
Output Guardrails
   |
   v
API/UI Response
```

## 4. Database and Storage Design

The initial system uses file-based storage rather than a centralized database.

- `data/raw/`: downloaded benchmark datasets.
- `data/processed/`: normalized training/evaluation assets.
- `models/`: fine-tuned model checkpoints and tokenizer artifacts.
- `data/vector_store/` planned: persisted FAISS index and chunk metadata.
- `data/prompt_registry.json` planned: prompt version metadata and promotion history.
- `results/` planned: evaluation and performance reports.

## 5. API Design

Planned primary endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Service health and model readiness |
| `/info` | GET | Runtime metadata and enabled capabilities |
| `/correct` | POST | Single-text correction |
| `/correct/batch` | POST | Batch correction |
| `/detect` | POST | Error detection only |
| `/knowledge/add` | POST | Add knowledge-base rules |
| `/knowledge/search` | GET | Semantic rule retrieval |
| `/prompts` | GET | List prompt versions |
| `/prompts/{version_id}` | GET | Get prompt details |
| `/prompts/{version_id}/promote` | POST | Promote prompt version |
| `/prompts/rollback` | POST | Roll back active prompt |
| `/evaluate` | POST | Compute evaluation metrics |

Request/response schemas will use Pydantic models in later sprints.

## 6. Error Handling Strategy

- Input validation failures return structured client errors with actionable messages.
- Model-load or inference failures are logged and translated into stable API error responses.
- Guardrail violations are categorized by severity to distinguish warnings from hard stops.
- Batch operations capture per-item failures without discarding successful items.
- CI checks prevent linting and test regressions from reaching `main`.

## 7. Security Considerations

- Prompt injection detection will be built into input validation.
- Unsafe or overly long payloads will be rejected before model execution.
- The initial release does not implement authentication, so it is intended for trusted environments or demos.
- Model artifacts and raw datasets remain outside version control.
- Future work may add rate limiting, authentication, audit logging, and secrets management.

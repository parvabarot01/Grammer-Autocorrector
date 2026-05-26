# API Reference

## Overview
This document covers the Sprint 2 Python utility interfaces that support the Grammar Autocorrector System. Full REST endpoint documentation will be expanded in Sprint 5 when the FastAPI service is implemented.

## Import Paths

```python
from src.utils.config import (
    APIConfig,
    Config,
    DataConfig,
    GuardrailsConfig,
    ModelConfig,
    RAGConfig,
    load_config,
)
from src.utils.evaluation import Evaluator
from src.utils.preprocessing import GrammarPreprocessor
```

## GrammarPreprocessor

### Summary
`GrammarPreprocessor` centralizes text cleanup, task-specific tokenization, sentence splitting, batch cleanup, and basic input validation.

### Example

```python
from transformers import AutoTokenizer

from src.utils.preprocessing import GrammarPreprocessor

preprocessor = GrammarPreprocessor()
tokenizer = AutoTokenizer.from_pretrained("t5-base")

cleaned = preprocessor.clean_text("  She go to school everyday.  ")
t5_inputs = preprocessor.tokenize_for_t5(cleaned, tokenizer, max_length=128)
sentences = preprocessor.split_into_sentences(cleaned)
```

### Public Methods

#### `clean_text(text: str) -> str`

Normalizes unicode, removes zero-width characters, collapses repeated whitespace, and trims surrounding whitespace.

Parameters:

| Name | Type | Description |
|------|------|-------------|
| `text` | `str` | Raw input text to normalize. |

Returns:

| Type | Description |
|------|-------------|
| `str` | Cleaned text ready for tokenization or validation. |

Example:

```python
preprocessor.clean_text("Cafe\u0301  \u200b is open.")
# "Café is open."
```

#### `tokenize_for_t5(text: str, tokenizer, max_length: int) -> dict`

Prepares text for sequence-to-sequence grammar correction by prefixing the input with `grammar: ` and tokenizing with padding and truncation.

Parameters:

| Name | Type | Description |
|------|------|-------------|
| `text` | `str` | Input text to tokenize. |
| `tokenizer` | `Any` | Hugging Face-compatible tokenizer instance. |
| `max_length` | `int` | Maximum token length after truncation. |

Returns:

| Type | Description |
|------|-------------|
| `dict` | Tokenizer payload containing `input_ids` and `attention_mask`. |

#### `tokenize_for_bert(text: str, tokenizer, max_length: int) -> dict`

Tokenizes text for BERT token classification. If the tokenizer does not supply `token_type_ids`, the method creates a zero-filled structure matching `input_ids`.

Parameters:

| Name | Type | Description |
|------|------|-------------|
| `text` | `str` | Input text to tokenize. |
| `tokenizer` | `Any` | Hugging Face-compatible tokenizer instance. |
| `max_length` | `int` | Maximum token length after truncation. |

Returns:

| Type | Description |
|------|-------------|
| `dict` | Tokenizer payload containing `input_ids`, `attention_mask`, and `token_type_ids`. |

#### `split_into_sentences(text: str) -> list[str]`

Splits normalized text into sentences using punctuation boundaries while protecting common abbreviations such as `Dr.` and `Mr.`.

Parameters:

| Name | Type | Description |
|------|------|-------------|
| `text` | `str` | Text block that may contain multiple sentences. |

Returns:

| Type | Description |
|------|-------------|
| `list[str]` | List of sentence strings. |

#### `batch_preprocess(texts: list[str]) -> list[str]`

Runs `clean_text` over a batch and filters out empty results.

Parameters:

| Name | Type | Description |
|------|------|-------------|
| `texts` | `list[str]` | Batch of raw texts. |

Returns:

| Type | Description |
|------|-------------|
| `list[str]` | Cleaned, non-empty text items. |

#### `validate_input(text: str, max_length: int = 1000) -> tuple[bool, str]`

Applies input sanity checks for emptiness, length, and non-printable characters.

Parameters:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | None | Raw user input. |
| `max_length` | `int` | `1000` | Maximum allowed character count after cleanup. |

Returns:

| Type | Description |
|------|-------------|
| `tuple[bool, str]` | Validation result as `(is_valid, error_message)`. |

## Evaluator

### Summary
`Evaluator` computes grammar correction metrics and can generate a markdown report for experiments or benchmark runs.

### Example

```python
from src.utils.evaluation import Evaluator

evaluator = Evaluator()
gleu = evaluator.compute_gleu(
    predictions=["She goes to school every day."],
    references=[["She goes to school every day."]],
)
```

### Public Methods

#### `compute_gleu(predictions: list[str], references: list[list[str]]) -> float`

Computes corpus-level GLEU with `sacrebleu`. Reference lists are expected in per-sample form and are internally transposed for corpus scoring.

#### `compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]`

Returns average `rouge1`, `rouge2`, and `rougeL` F-measure scores using `rouge-score`.

#### `compute_m2_score(original: list[str], corrected: list[str], gold_edits: list[list[str]]) -> dict[str, float]`

Builds a lightweight M2-style comparison from predicted edits and gold edit annotations, returning:

| Key | Description |
|-----|-------------|
| `precision` | Fraction of predicted edits that match gold edits. |
| `recall` | Fraction of gold edits recovered by predictions. |
| `f05` | F0.5 score favoring precision, common in GEC evaluation. |

#### `compute_exact_match(predictions: list[str], references: list[str]) -> float`

Computes case-insensitive exact match accuracy.

#### `evaluate_batch(model_fn, dataset, batch_size: int = 32) -> dict`

Runs a model function over a dataset and returns a combined metrics dictionary. The dataset records may use any of the following keys:

- Original text: `original`, `source`, `input`, `sentence`, `text`
- References: `references`, `reference`, `targets`, `target`, `corrections`
- Optional gold edits: `gold_edits`

Expected output dictionary keys:

| Key | Description |
|-----|-------------|
| `samples_evaluated` | Number of evaluated items. |
| `gleu` | Corpus-level GLEU score. |
| `rouge` | Nested ROUGE metrics dictionary. |
| `exact_match` | Exact match accuracy. |
| `m2` | Nested M2-style precision/recall/F0.5 dictionary. |
| `model_name` | Inferred model function name. |

#### `generate_evaluation_report(metrics: dict, output_path: str) -> None`

Writes a markdown report containing:

- UTC timestamp
- model name
- flattened metrics table

Example:

```python
metrics = {
    "model_name": "demo-corrector",
    "gleu": 0.72,
    "rouge": {"rouge1": 0.80, "rouge2": 0.71, "rougeL": 0.78},
}
evaluator.generate_evaluation_report(metrics, "results/evaluation_report.md")
```

## Configuration Dataclasses

### `ModelConfig`

| Field | Type | Default |
|-------|------|---------|
| `t5_model_name` | `str` | `"t5-base"` |
| `bert_model_name` | `str` | `"bert-base-uncased"` |
| `max_length` | `int` | `128` |
| `num_beams` | `int` | `4` |
| `learning_rate` | `float` | `3e-4` |
| `batch_size` | `int` | `16` |
| `epochs` | `int` | `5` |

### `DataConfig`

| Field | Type | Default |
|-------|------|---------|
| `conll2014_name` | `str` | `"conll2014"` |
| `bea2019_name` | `str` | `"jfleg"` |
| `jfleg_name` | `str` | `"jfleg"` |
| `train_split` | `float` | `0.8` |
| `val_split` | `float` | `0.1` |
| `test_split` | `float` | `0.1` |
| `raw_data_path` | `Path` | `BASE_DIR / "data/raw"` |
| `processed_data_path` | `Path` | `BASE_DIR / "data/processed"` |
| `sample_data_path` | `Path` | `BASE_DIR / "data/sample"` |

### `RAGConfig`

| Field | Type | Default |
|-------|------|---------|
| `embedding_model_name` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` |
| `vector_store_path` | `Path` | `BASE_DIR / "data/vector_store"` |
| `top_k` | `int` | `5` |
| `chunk_size` | `int` | `512` |
| `chunk_overlap` | `int` | `50` |

### `APIConfig`

| Field | Type | Default |
|-------|------|---------|
| `host` | `str` | `"0.0.0.0"` |
| `port` | `int` | `8000` |
| `max_concurrent_requests` | `int` | `100` |

### `GuardrailsConfig`

| Field | Type | Default |
|-------|------|---------|
| `toxicity_threshold` | `float` | `0.8` |
| `max_input_length` | `int` | `1000` |

### `Config`

| Field | Type | Description |
|-------|------|-------------|
| `model` | `ModelConfig` | Model and training defaults. |
| `data` | `DataConfig` | Dataset names, splits, and file-system paths. |
| `rag` | `RAGConfig` | Retrieval and embedding settings. |
| `api` | `APIConfig` | API host, port, and concurrency settings. |
| `guardrails` | `GuardrailsConfig` | Safety thresholds and input limits. |

## `load_config(dotenv_path: str | Path | None = None) -> Config`

Loads `.env` values and applies sensible defaults when variables are not defined.

Common environment variables:

| Environment Variable | Maps To |
|----------------------|---------|
| `T5_MODEL_NAME` | `ModelConfig.t5_model_name` |
| `BERT_MODEL_NAME` | `ModelConfig.bert_model_name` |
| `MODEL_MAX_LENGTH` | `ModelConfig.max_length` |
| `MODEL_NUM_BEAMS` | `ModelConfig.num_beams` |
| `MODEL_LEARNING_RATE` | `ModelConfig.learning_rate` |
| `MODEL_BATCH_SIZE` | `ModelConfig.batch_size` |
| `MODEL_EPOCHS` | `ModelConfig.epochs` |
| `DATA_TRAIN_SPLIT` | `DataConfig.train_split` |
| `DATA_VAL_SPLIT` | `DataConfig.val_split` |
| `DATA_TEST_SPLIT` | `DataConfig.test_split` |
| `RAW_DATA_PATH` | `DataConfig.raw_data_path` |
| `PROCESSED_DATA_PATH` | `DataConfig.processed_data_path` |
| `RAG_VECTOR_STORE_PATH` | `RAGConfig.vector_store_path` |
| `API_HOST` | `APIConfig.host` |
| `API_PORT` | `APIConfig.port` |
| `GUARDRAILS_TOXICITY_THRESHOLD` | `GuardrailsConfig.toxicity_threshold` |
| `GUARDRAILS_MAX_INPUT_LENGTH` | `GuardrailsConfig.max_input_length` |

## Planned REST Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Service readiness and model status |
| `/info` | GET | Capability and runtime metadata |
| `/correct` | POST | Single-text correction |
| `/correct/batch` | POST | Batch correction |
| `/detect` | POST | Error detection only |
| `/knowledge/add` | POST | Add grammar rules to the knowledge base |
| `/knowledge/search` | GET | Search retrieved rules or context |
| `/prompts` | GET | List prompt versions |
| `/prompts/{version_id}` | GET | Retrieve a prompt version |
| `/prompts/{version_id}/promote` | POST | Promote a prompt version |
| `/prompts/rollback` | POST | Revert to the previous active prompt |
| `/evaluate` | POST | Compute evaluation metrics |

# Grammar Autocorrector System

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-ci%20configured-brightgreen.svg)](.github/workflows/ci.yml)

Production-oriented NLP grammar correction platform combining transformer-based correction, token-level error detection, retrieval-augmented prompting, and responsible AI guardrails. The repository is structured to support research experimentation, API serving, and an interactive UI from a single codebase.

This Sprint 1 scaffold establishes the engineering foundation for later sprints that will implement T5, BERT, RNN, RAG, FastAPI, and Gradio components.

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Datasets](#datasets)
- [Evaluation Results](#evaluation-results)
- [API Usage](#api-usage)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Features

- T5 sequence-to-sequence grammar correction pipeline for fluent sentence rewriting.
- BERT token-level grammar error detection for locating likely error spans.
- RNN baseline model for benchmarking neural performance against modern transformers.
- Retrieval-Augmented Generation workflow using embeddings and FAISS-backed rule lookup.
- Prompt versioning for controlled experimentation and production rollout.
- Responsible AI guardrails including validation, toxicity checks, and bias-aware review.
- FastAPI backend design for inference, evaluation, and prompt/knowledge management.
- Gradio web interface plan for single-text, batch, and evaluation workflows.
- Test-first project layout with unit, integration, and performance test planning.

## Architecture Overview

```text
Input Text
    |
    v
+-----------------------+
| Input Validation      |
| and Guardrails        |
+-----------+-----------+
            |
            v
+-----------------------+
| BERT Error Detector   |
| Detects error spans   |
+-----------+-----------+
            |
            v
+-----------------------+        +------------------------+
| Mode Selection        |<------>| Prompt Versioning      |
| T5 / RNN / RAG        |        | Active prompt template |
+-----------+-----------+        +------------------------+
            |
            +------------------------------+
            |                              |
            v                              v
+-----------------------+        +------------------------+
| T5 Corrector          |        | RAG Pipeline           |
| Seq2seq correction    |        | Embeddings + FAISS     |
+-----------+-----------+        +-----------+------------+
            |                                |
            +---------------+----------------+
                            |
                            v
                  +-----------------------+
                  | Output Guardrails     |
                  | Safety + consistency  |
                  +-----------+-----------+
                              |
                              v
                        Corrected Output
```

## Installation

1. Create and activate a virtual environment.
2. Install Python dependencies.
3. Use the provided `Makefile` for linting, testing, and future training commands.

```bash
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

The example below shows the planned usage interface that later sprints will implement:

```python
from src.models.t5_corrector import T5GrammarCorrector

model = T5GrammarCorrector(model_name="t5-base")
model.load_model()
print(model.correct("She go to school everyday."))
```

## Project Structure

```text
grammar-autocorrector/
|- README.md                  # Project overview and onboarding
|- CHANGELOG.md               # Release history
|- LICENSE                    # License terms
|- requirements.txt           # Python dependencies
|- Makefile                   # Common development commands
|- docs/                      # Requirements, design, architecture, testing, guides
|- src/                       # Application source code
|  |- models/                 # T5, BERT, and RNN model modules
|  |- pipeline/               # RAG, prompt versioning, and guardrails
|  |- api/                    # FastAPI backend
|  |- utils/                  # Shared utilities and configuration
|  \- ui/                     # Gradio frontend
|- data/                      # Raw, processed, and sample datasets
|- models/                    # Trained checkpoints (gitignored)
|- notebooks/                 # Exploratory and demo notebooks
|- tests/                     # Unit and integration tests
|- scripts/                   # Training, download, and evaluation scripts
\- .github/workflows/         # CI and lint automation
```

## Models Used

- `T5`: Primary sequence-to-sequence grammar correction model that rewrites erroneous text into corrected text.
- `BERT`: Token classification model that highlights likely error spans for analysis and explainability.
- `RNN Baseline`: Bidirectional recurrent encoder-decoder baseline used to benchmark simpler architectures.

## Datasets

Planned benchmark and training references:

- `CoNLL-2014 Shared Task`: Standard grammatical error correction benchmark.
- `BEA-2019`: Broad evaluation benchmark for grammar correction systems.
- `JFLEG`: Fluency-oriented benchmark often used for grammar correction experimentation.

Raw and processed data directories are scaffolded now and will be populated in later sprints.

## Evaluation Results

| Model | Precision | Recall | F1 | Accuracy |
|-------|-----------|--------|----|----------|
| T5 Corrector | TBD | TBD | TBD | TBD |
| BERT Detector | TBD | TBD | TBD | TBD |
| RNN Baseline | TBD | TBD | TBD | TBD |
| RAG-Enhanced Pipeline | TBD | TBD | TBD | TBD |

## API Usage

Planned REST inference example:

```bash
curl -X POST "http://localhost:8000/correct" \
  -H "Content-Type: application/json" \
  -d '{"text":"She go to school everyday.","mode":"auto","num_beams":4}'
```

Planned Python client example:

```python
import requests

response = requests.post(
    "http://localhost:8000/correct",
    json={"text": "She go to school everyday.", "mode": "auto", "num_beams": 4},
    timeout=30,
)
print(response.json())
```

## Contributing

1. Create a feature branch from `main`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `make lint` and `make test` before opening a pull request.
4. Document architectural or behavioral changes in `docs/` and `CHANGELOG.md`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

Parva Barot

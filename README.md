# Grammar Autocorrector System

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen.svg)]()

> An NLP-based grammar correction system built around fine-tuned T5, BERT error
> detection, retrieval-augmented prompting, and responsible AI guardrails. The
> repository includes the training, serving, benchmarking, and UI stack, with
> heuristic fallbacks so the API remains runnable before local model weights are
> added.

## Features

- **T5 Sequence-to-Sequence Correction** for fluent grammar rewriting with beam search support
- **BERT Token-level Error Detection** for error-span highlighting and explainability
- **RNN Baseline** with bidirectional LSTM and Bahdanau attention for benchmarking
- **RAG Pipeline** with FAISS-compatible retrieval over grammar-rule knowledge
- **Prompt Versioning** with semantic versioning, promotion, rollback, and metric tracking
- **Responsible AI Guardrails** for input validation, toxicity checks, bias checks, and output validation
- **FastAPI Backend** with correction, batch, evaluation, prompt, knowledge, and benchmark endpoints
- **Interactive Gradio UI** with tabs for correction, batch processing, detection, prompt management, and evaluation
- **Docker-ready Deployment** for API and UI services
- **Testing and Profiling Tooling** with integration tests, coverage reporting, and performance profiling

## Architecture

```text
Input Text
    |
    v
+---------------------+    <- Length check, toxicity, prompt injection defense
|  Input Guardrails   |
+----------+----------+
           |
     +-----v------+
     | BERT Detect |    <- Token-level error classification
     +-----+------+
           | errors detected?
     +-----v----------------------+
     |  Correction Mode Selection |
     |  - T5 (beam search)        |
     |  - RAG + T5                |
     +-----+----------------------+
           |
     +-----v------+    <- FAISS vector search -> grammar rules
     | RAG Lookup |
     +-----+------+
           |
     +-----v------+    <- Fine-tuned seq2seq or local fallback
     | T5 Correct |
     +-----+------+
           |
+----------v----------+    <- Length ratio, meaning preservation
|  Output Guardrails  |
+----------+----------+
           |
           v
    Corrected Text
```

## Quick Start

```bash
git clone https://github.com/parvabarot/grammar-autocorrector
cd grammar-autocorrector
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```python
from src.pipeline.correction_pipeline import CorrectionPipeline
from src.utils.config import load_config

pipeline = CorrectionPipeline(load_config())
pipeline.load_all()
result = pipeline.correct("She go to school everyday.")
print(result.corrected)
```

## Docker

```bash
docker-compose up --build
```

- API docs: `http://localhost:8000/docs`
- UI: `http://localhost:7860`

## Project Structure

```text
grammar-autocorrector/
|-- README.md
|-- CHANGELOG.md
|-- LICENSE
|-- .env.example
|-- requirements.txt
|-- setup.py
|-- pyproject.toml
|-- Makefile
|-- Dockerfile
|-- docker-compose.yml
|-- docs/                    # SRS, SDD, architecture, API, user, deployment, test docs
|-- src/
|   |-- models/             # T5, BERT, and RNN model implementations
|   |-- pipeline/           # Unified correction pipeline, RAG, prompts, guardrails
|   |-- api/                # FastAPI application, routes, and schemas
|   |-- utils/              # Config, preprocessing, and evaluation utilities
|   `-- ui/                 # Gradio application
|-- data/                   # Raw, processed, sample, registry, and vector-store inputs
|-- models/                 # Local checkpoints (gitignored)
|-- notebooks/              # Exploration, fine-tuning, evaluation, and demo notebooks
|-- results/                # Evaluation and profiling outputs
|-- tests/                  # Unit and integration test suites
`-- scripts/                # Training, evaluation, download, and profiling entrypoints
```

## Models and Data

- **T5** is the primary grammar correction model and main serving target
- **BERT** performs token-level detection to decide when correction is needed and to surface error spans
- **RNN Baseline** provides a classical seq2seq benchmark for comparison
- **CoNLL-2014**, **BEA-2019-style evaluation**, and **JFLEG** are the intended benchmark families
- The repository does **not** ship trained weights. Add local checkpoints or run the training scripts first for real benchmark scoring.

## Evaluation and Benchmarking

Two result artifacts are tracked in the repository structure:

- [results/evaluation_report.md](/C:/Users/parva/OneDrive/Desktop/Project/Grammer Autocorrector/results/evaluation_report.md)
- [results/performance_profile.md](/C:/Users/parva/OneDrive/Desktop/Project/Grammer Autocorrector/results/performance_profile.md)

Quality metrics are produced after training and evaluation runs. The committed
repository intentionally avoids publishing fabricated model scores before
checkpoints exist locally.

Representative commands:

```bash
python scripts/evaluate.py
python scripts/profile_pipeline.py
```

## Running the Stack

```bash
make test
make lint
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
python src/ui/gradio_app.py
```

## Documentation

- [Software Requirements Specification](docs/SRS.md)
- [Software Design Document](docs/SDD.md)
- [Architecture Decision Records](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [User Guide](docs/user_guide.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Test Plan](docs/testing_plan.md)
- [Known Issues](docs/known_issues.md)

## Author

**Parva Barot** - Software Engineer  
MS Information Technology, Arizona State University (2025)  
[LinkedIn](https://linkedin.com/in/parvabarot) | [GitHub](https://github.com/parvabarot01)

## License

MIT License. See [LICENSE](LICENSE) for details.

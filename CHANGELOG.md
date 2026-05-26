# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.0.0] - Sprint 6 (Production Release)
### Added
- CorrectionPipeline unified orchestration class
- Benchmark endpoint and performance profiling script
- End-to-end integration tests
- Performance profiling results and results directory
- Final polished README
- Complete user guide
- Environment example and known issues documentation

### Changed
- API wired to CorrectionPipeline as the single dependency
- Coverage gate raised above 80% for the runnable runtime layers

### Fixed
- Sprint integration gaps between API, RAG, prompt management, and guardrails
- Output validation and batch-failure handling edge cases

## [0.5.0] - Sprint 5
### Added
- FastAPI REST API with health, correction, detection, knowledge base, prompt management, and evaluation endpoints
- Pydantic request and response schemas with validation and OpenAPI examples
- CORS, request timing, and request ID middleware
- Gradio web interface with six workflow tabs
- API integration tests using FastAPI TestClient and mock runtime dependencies
- Dockerfile and docker-compose.yml for API and UI deployment
- Complete API reference and deployment guide

## [0.4.0] - Sprint 4
### Added
- GrammarRAGPipeline with FAISS-compatible vector indexing, embedding search, and prompt augmentation
- 30-rule grammar knowledge base
- PromptVersionManager with semver versioning, promotion, rollback, and metrics updates
- 3 pre-registered prompt versions for simple, RAG-augmented, and chain-of-thought prompting
- GrammarGuardrails with input/output validation, toxicity filtering, and bias detection
- RAG pipeline demo notebook
- Unit tests for guardrails
- Integration tests for the RAG, prompt, and guardrail pipeline

## [0.3.0] - Sprint 3
### Added
- T5GrammarCorrector with fine-tuning, beam search decoding, and batch correction
- BERTGrammarDetector with token-level error detection and confidence scoring
- RNNSeq2Seq baseline with Bahdanau attention
- Training scripts for T5 and BERT with argparse interfaces
- Evaluation script generating markdown reports
- T5 fine-tuning notebook and BERT evaluation notebook
- Unit tests for model modules using mocking

## [0.2.0] - Sprint 2
### Added
- Data pipeline and download scripts (CoNLL-2014, JFLEG datasets)
- GrammarPreprocessor class with full text cleaning and tokenization
- Evaluator class with GLEU, ROUGE, M2, exact match metrics
- Configuration management with dataclasses
- Data exploration notebook
- Unit tests for preprocessing (>90% coverage)
- 20 sample sentences with annotated errors

## [0.1.0] - Sprint 1
### Added
- Project scaffolding and directory structure
- Software Requirements Specification (SRS)
- Software Design Document (SDD)
- Architecture Decision Records (ADRs)
- Test Plan
- Initial README

# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

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

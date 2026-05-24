# API Reference

## Overview
This document tracks the planned public interfaces for the Grammar Autocorrector System. Detailed Python utility references will be added in Sprint 2, and complete REST endpoint documentation will be expanded in Sprint 5.

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

## Planned Python Modules

- `src.utils.preprocessing`: text cleaning, tokenization, validation helpers
- `src.utils.evaluation`: benchmark metric utilities and reporting
- `src.utils.config`: environment-aware configuration dataclasses
- `src.models.t5_corrector`: seq2seq grammar correction interface
- `src.models.bert_detector`: token-level error detection interface
- `src.pipeline.rag_pipeline`: retrieval and prompt augmentation interface

## Versioning
The API will begin as an unversioned v1 service and is expected to adopt explicit route versioning when breaking changes become likely.

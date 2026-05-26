"""FastAPI route definitions for the Grammar Autocorrector service."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.api.runtime import APIRuntime, utc_timestamp
from src.api.schemas import (
    BatchCorrectionRequest,
    BatchCorrectionResponse,
    CorrectionRequest,
    CorrectionResponse,
    DetectionRequest,
    DetectionResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    InfoResponse,
    KnowledgeAddRequest,
    KnowledgeAddResponse,
    KnowledgeSearchResponse,
    PromptListResponse,
    PromptPromoteResponse,
    PromptRollbackResponse,
    PromptVersionResponse,
)
from src.pipeline import GuardrailViolation

LOGGER = logging.getLogger(__name__)
router = APIRouter()


def get_runtime(request: Request) -> APIRuntime:
    """Return the initialized API runtime from application state."""

    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("API runtime has not been initialized.")
    return runtime


def _request_id(request: Request) -> str:
    """Return the request id assigned by middleware."""

    return getattr(request.state, "request_id", "unknown")


def _api_error(status_code: int, message: str) -> HTTPException:
    """Create a uniform HTTP exception for route-level validation failures."""

    return HTTPException(status_code=status_code, detail=message)


HEALTH_OPENAPI = {
    "x-example-response": {
        "status": "ok",
        "models_loaded": False,
        "version": "1.0.0",
        "timestamp": "2026-05-25T12:00:00+00:00",
    }
}
INFO_OPENAPI = {
    "x-example-response": {
        "models": ["T5GrammarCorrector", "BERTGrammarDetector", "GrammarRAGPipeline"],
        "prompt_version": "v1.1.0",
        "capabilities": [
            "single-correction",
            "batch-correction",
            "error-detection",
            "knowledge-search",
            "prompt-management",
            "evaluation",
        ],
    }
}
CORRECT_OPENAPI = {
    "requestBody": {
        "content": {
            "application/json": {
                "example": {
                    "text": "She go to school everyday.",
                    "mode": "auto",
                    "num_beams": 4,
                    "return_detected_errors": True,
                    "prompt_version": "v1.1.0",
                }
            }
        }
    }
}
BATCH_OPENAPI = {
    "requestBody": {
        "content": {
            "application/json": {
                "example": {
                    "texts": ["She go to school everyday.", "He have a apple."],
                    "mode": "auto",
                    "batch_size": 8,
                }
            }
        }
    }
}
DETECT_OPENAPI = {
    "requestBody": {
        "content": {
            "application/json": {"example": {"text": "She go to school everyday."}}
        }
    }
}
KNOWLEDGE_ADD_OPENAPI = {
    "requestBody": {
        "content": {
            "application/json": {
                "example": {
                    "rules": [
                        "Use 'an' before a vowel sound.",
                        "A singular subject normally takes a singular verb.",
                    ]
                }
            }
        }
    }
}
KNOWLEDGE_SEARCH_OPENAPI = {
    "x-query-example": {"query": "She go to school everyday.", "top_k": 3}
}
PROMPTS_OPENAPI = {"x-example-response": {"active_version": "v1.1.0"}}
EVALUATE_OPENAPI = {
    "requestBody": {
        "content": {
            "application/json": {
                "example": {
                    "predictions": ["She goes to school every day."],
                    "references": ["She goes to school every day."],
                    "metrics": ["gleu", "rouge", "exact_match"],
                }
            }
        }
    }
}


@router.get("/health", response_model=HealthResponse, openapi_extra=HEALTH_OPENAPI)
def health(request: Request, runtime: APIRuntime = Depends(get_runtime)) -> Any:
    """Return service health and model loading status."""

    return {
        "status": "ok",
        "models_loaded": runtime.models_loaded,
        "version": request.app.version,
        "timestamp": utc_timestamp(),
    }


@router.get("/info", response_model=InfoResponse, openapi_extra=INFO_OPENAPI)
def info(runtime: APIRuntime = Depends(get_runtime)) -> Any:
    """Return service metadata, active prompt version, and capabilities."""

    return {
        "models": [
            "T5GrammarCorrector",
            "BERTGrammarDetector",
            "GrammarRAGPipeline",
            "GrammarGuardrails",
        ],
        "prompt_version": runtime.prompt_manager.get_active_prompt().version_id,
        "capabilities": [
            "single-correction",
            "batch-correction",
            "error-detection",
            "knowledge-search",
            "prompt-management",
            "evaluation",
        ],
    }


@router.post(
    "/correct",
    response_model=CorrectionResponse,
    openapi_extra=CORRECT_OPENAPI,
)
def correct_text(
    payload: CorrectionRequest,
    request: Request,
    runtime: APIRuntime = Depends(get_runtime),
) -> Any:
    """Correct a single input string."""

    if not payload.text.strip():
        raise _api_error(400, "Input text cannot be empty.")

    try:
        result = runtime.correct(
            text=payload.text,
            mode=payload.mode,
            num_beams=payload.num_beams,
            return_detected_errors=payload.return_detected_errors,
            prompt_version=payload.prompt_version,
        )
    except GuardrailViolation as exc:
        raise _api_error(422, str(exc)) from exc
    except KeyError as exc:
        raise _api_error(400, str(exc)) from exc
    except ValueError as exc:
        raise _api_error(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive server path
        LOGGER.exception("Correction failed.")
        raise _api_error(500, str(exc)) from exc

    serialized = runtime.serialize(result)
    serialized["request_id"] = _request_id(request)
    serialized.pop("prompt_version", None)
    return serialized


@router.post(
    "/correct/batch",
    response_model=BatchCorrectionResponse,
    openapi_extra=BATCH_OPENAPI,
)
def correct_batch(
    payload: BatchCorrectionRequest,
    runtime: APIRuntime = Depends(get_runtime),
) -> Any:
    """Correct a batch of texts with per-item fault tolerance."""

    started = perf_counter()
    try:
        results = runtime.correct_batch(
            texts=payload.texts,
            mode=payload.mode,
            batch_size=payload.batch_size,
        )
    except ValueError as exc:
        raise _api_error(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive server path
        LOGGER.exception("Batch correction failed.")
        raise _api_error(500, str(exc)) from exc

    return {
        "results": runtime.serialize(results),
        "total_processed": len(results),
        "processing_time_ms": round((perf_counter() - started) * 1000, 3),
    }


@router.post(
    "/detect",
    response_model=DetectionResponse,
    openapi_extra=DETECT_OPENAPI,
)
def detect_errors(
    payload: DetectionRequest,
    runtime: APIRuntime = Depends(get_runtime),
) -> Any:
    """Detect token-level grammar errors without applying correction."""

    if not payload.text.strip():
        raise _api_error(400, "Input text cannot be empty.")

    try:
        return runtime.serialize(runtime.detect(payload.text))
    except GuardrailViolation as exc:
        raise _api_error(422, str(exc)) from exc
    except ValueError as exc:
        raise _api_error(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive server path
        LOGGER.exception("Error detection failed.")
        raise _api_error(500, str(exc)) from exc


@router.post(
    "/knowledge/add",
    response_model=KnowledgeAddResponse,
    openapi_extra=KNOWLEDGE_ADD_OPENAPI,
)
def add_knowledge(
    payload: KnowledgeAddRequest,
    runtime: APIRuntime = Depends(get_runtime),
) -> Any:
    """Add rules to the grammar knowledge base."""

    try:
        return runtime.add_knowledge_rules(payload.rules)
    except ValueError as exc:
        raise _api_error(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive server path
        LOGGER.exception("Knowledge base update failed.")
        raise _api_error(500, str(exc)) from exc


@router.get(
    "/knowledge/search",
    response_model=KnowledgeSearchResponse,
    openapi_extra=KNOWLEDGE_SEARCH_OPENAPI,
)
def search_knowledge(
    query: str = Query(
        ..., description="Search query for grammar rules.", min_length=1
    ),
    top_k: int = Query(
        5, description="Maximum number of rules to retrieve.", ge=1, le=20
    ),
    runtime: APIRuntime = Depends(get_runtime),
) -> Any:
    """Search the grammar-rule knowledge base."""

    try:
        return runtime.serialize(runtime.search_knowledge(query=query, top_k=top_k))
    except ValueError as exc:
        raise _api_error(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive server path
        LOGGER.exception("Knowledge search failed.")
        raise _api_error(500, str(exc)) from exc


@router.get(
    "/prompts", response_model=PromptListResponse, openapi_extra=PROMPTS_OPENAPI
)
def list_prompts(runtime: APIRuntime = Depends(get_runtime)) -> Any:
    """List registered prompts and the active version."""

    return runtime.serialize(runtime.list_prompt_versions())


@router.get(
    "/prompts/{version_id}",
    response_model=PromptVersionResponse,
    openapi_extra={"x-path-example": {"version_id": "v1.1.0"}},
)
def get_prompt(version_id: str, runtime: APIRuntime = Depends(get_runtime)) -> Any:
    """Return a single prompt version by id."""

    try:
        return runtime.serialize(runtime.get_prompt_version(version_id))
    except KeyError as exc:
        raise _api_error(400, str(exc)) from exc


@router.post(
    "/prompts/{version_id}/promote",
    response_model=PromptPromoteResponse,
    openapi_extra={"x-path-example": {"version_id": "v2.0.0"}},
)
def promote_prompt(version_id: str, runtime: APIRuntime = Depends(get_runtime)) -> Any:
    """Promote a prompt version to active status."""

    try:
        return runtime.promote_prompt(version_id)
    except KeyError as exc:
        raise _api_error(400, str(exc)) from exc


@router.post(
    "/prompts/rollback",
    response_model=PromptRollbackResponse,
    openapi_extra={"x-example-response": {"rolled_back_to": "v1.1.0"}},
)
def rollback_prompt(runtime: APIRuntime = Depends(get_runtime)) -> Any:
    """Rollback to the previously active prompt version."""

    return runtime.rollback_prompt()


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    openapi_extra=EVALUATE_OPENAPI,
)
def evaluate(
    payload: EvaluateRequest,
    runtime: APIRuntime = Depends(get_runtime),
) -> Any:
    """Evaluate predictions against references with selected metrics."""

    try:
        return runtime.evaluate_metrics(
            predictions=payload.predictions,
            references=payload.references,
            metrics=payload.metrics,
        )
    except ValueError as exc:
        raise _api_error(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive server path
        LOGGER.exception("Evaluation failed.")
        raise _api_error(500, str(exc)) from exc

"""FastAPI route definitions for the Grammar Autocorrector service."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict
from difflib import SequenceMatcher
from time import perf_counter
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.api.schemas import (
    BatchCorrectionRequest,
    BatchCorrectionResponse,
    BenchmarkReportResponse,
    BenchmarkRequest,
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
    PublicCorrectionRequest,
    PublicCorrectionResponse,
)
from src.pipeline import CorrectionPipeline, GuardrailViolation

LOGGER = logging.getLogger(__name__)
router = APIRouter()


def get_pipeline(request: Request) -> CorrectionPipeline:
    """Return the initialized correction pipeline from application state."""

    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("Correction pipeline has not been initialized.")
    return pipeline


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
        "timestamp": "2026-05-26T12:00:00+00:00",
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
            "benchmarking",
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
PUBLIC_CORRECT_OPENAPI = {
    "requestBody": {
        "content": {"application/json": {"example": {"text": "She go to school."}}}
    },
    "x-example-response": {
        "original_text": "She go to school.",
        "corrected_text": "She goes to school.",
        "changes": [
            {
                "before": "go",
                "after": "goes",
                "explanation": "Grammar corrected for clarity and correctness.",
            }
        ],
        "summary": "1 grammar issue corrected.",
        "success": True,
    },
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
BENCHMARK_OPENAPI = {
    "requestBody": {
        "content": {
            "application/json": {
                "example": {
                    "test_pairs": [
                        {
                            "original": "She go to school everyday.",
                            "reference": "She goes to school every day.",
                        },
                        {
                            "original": "He have a apple.",
                            "reference": "He has an apple.",
                        },
                    ],
                    "max_samples": 2,
                }
            }
        }
    }
}


def _format_change_fragment(tokens: List[str]) -> str:
    """Join diff tokens into a compact user-facing text fragment."""

    text = " ".join(tokens)
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def _build_public_changes(original: str, corrected: str) -> List[Dict[str, str]]:
    """Build a small public-safe change list from corrected text."""

    token_pattern = r"\w+(?:'\w+)?|[^\w\s]"
    original_tokens = re.findall(token_pattern, original)
    corrected_tokens = re.findall(token_pattern, corrected)
    matcher = SequenceMatcher(
        a=[token.casefold() for token in original_tokens],
        b=[token.casefold() for token in corrected_tokens],
    )

    changes: List[Dict[str, str]] = []
    for (
        operation,
        original_start,
        original_end,
        corrected_start,
        corrected_end,
    ) in matcher.get_opcodes():
        if operation == "equal":
            continue
        changes.append(
            {
                "before": _format_change_fragment(
                    original_tokens[original_start:original_end]
                ),
                "after": _format_change_fragment(
                    corrected_tokens[corrected_start:corrected_end]
                ),
                "explanation": "Grammar corrected for clarity and correctness.",
            }
        )
    return changes


def _public_summary(changes: List[Dict[str, str]]) -> str:
    """Return a concise public correction summary."""

    issue_count = len(changes)
    if issue_count == 0:
        return "No grammar issues found."
    if issue_count == 1:
        return "1 grammar issue corrected."
    return f"{issue_count} grammar issues corrected."


@router.get("/health", response_model=HealthResponse, openapi_extra=HEALTH_OPENAPI)
def health(
    request: Request,
    pipeline: CorrectionPipeline = Depends(get_pipeline),
) -> Any:
    """Return service health and model loading status."""

    return {
        "status": "ok",
        "models_loaded": pipeline.models_loaded,
        "version": request.app.version,
        "timestamp": pipeline.utcnow(),
    }


@router.get("/info", response_model=InfoResponse, openapi_extra=INFO_OPENAPI)
def info(pipeline: CorrectionPipeline = Depends(get_pipeline)) -> Any:
    """Return service metadata, active prompt version, and capabilities."""

    return {
        "models": [
            "T5GrammarCorrector",
            "BERTGrammarDetector",
            "GrammarRAGPipeline",
            "GrammarGuardrails",
        ],
        "prompt_version": pipeline.prompt_manager.get_active_prompt().version_id,
        "capabilities": [
            "single-correction",
            "batch-correction",
            "error-detection",
            "knowledge-search",
            "prompt-management",
            "evaluation",
            "benchmarking",
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
    pipeline: CorrectionPipeline = Depends(get_pipeline),
) -> Any:
    """Correct a single input string."""

    if not payload.text.strip():
        raise _api_error(400, "Input text cannot be empty.")

    try:
        result = pipeline.correct(
            text=payload.text,
            mode=payload.mode,
            num_beams=payload.num_beams,
            return_errors=payload.return_detected_errors,
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

    serialized = pipeline.serialize(result)
    serialized["request_id"] = _request_id(request)
    return serialized


@router.post(
    "/public/correct",
    response_model=PublicCorrectionResponse,
    openapi_extra=PUBLIC_CORRECT_OPENAPI,
)
def public_correct_text(
    payload: PublicCorrectionRequest,
    request: Request,
    pipeline: CorrectionPipeline = Depends(get_pipeline),
) -> Any:
    """Correct text through a deliberately minimal public response contract."""

    if not request.app.state.config.api.enable_public_api:
        raise _api_error(404, "Public correction API is disabled.")

    try:
        result = pipeline.serialize(pipeline.correct(text=payload.text, mode="auto"))
    except GuardrailViolation as exc:
        raise _api_error(422, str(exc)) from exc
    except ValueError as exc:
        raise _api_error(
            400, "Unable to process text. Please check your input."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive server path
        LOGGER.exception("Public correction failed.")
        raise _api_error(500, "Unable to correct text right now.") from exc

    original_text = str(result["original"])
    corrected_text = str(result["corrected"])
    changes = _build_public_changes(original_text, corrected_text)
    return {
        "original_text": original_text,
        "corrected_text": corrected_text,
        "changes": changes,
        "summary": _public_summary(changes),
        "success": True,
    }


@router.post(
    "/correct/batch",
    response_model=BatchCorrectionResponse,
    openapi_extra=BATCH_OPENAPI,
)
def correct_batch(
    payload: BatchCorrectionRequest,
    pipeline: CorrectionPipeline = Depends(get_pipeline),
) -> Any:
    """Correct a batch of texts with per-item fault tolerance."""

    started = perf_counter()
    try:
        results = pipeline.correct_batch(
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
        "results": pipeline.serialize(results),
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
    pipeline: CorrectionPipeline = Depends(get_pipeline),
) -> Any:
    """Detect token-level grammar errors without applying correction."""

    if not payload.text.strip():
        raise _api_error(400, "Input text cannot be empty.")

    try:
        return pipeline.serialize(pipeline.detect(payload.text))
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
    pipeline: CorrectionPipeline = Depends(get_pipeline),
) -> Any:
    """Add rules to the grammar knowledge base."""

    try:
        return pipeline.add_knowledge_rules(payload.rules)
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
    pipeline: CorrectionPipeline = Depends(get_pipeline),
) -> Any:
    """Search the grammar-rule knowledge base."""

    try:
        return pipeline.serialize(pipeline.search_knowledge(query=query, top_k=top_k))
    except ValueError as exc:
        raise _api_error(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive server path
        LOGGER.exception("Knowledge search failed.")
        raise _api_error(500, str(exc)) from exc


@router.get(
    "/prompts", response_model=PromptListResponse, openapi_extra=PROMPTS_OPENAPI
)
def list_prompts(pipeline: CorrectionPipeline = Depends(get_pipeline)) -> Any:
    """List registered prompts and the active version."""

    return pipeline.serialize(pipeline.list_prompt_versions())


@router.get(
    "/prompts/{version_id}",
    response_model=PromptVersionResponse,
    openapi_extra={"x-path-example": {"version_id": "v1.1.0"}},
)
def get_prompt(
    version_id: str,
    pipeline: CorrectionPipeline = Depends(get_pipeline),
) -> Any:
    """Return a single prompt version by id."""

    try:
        return pipeline.serialize(pipeline.get_prompt_version(version_id))
    except KeyError as exc:
        raise _api_error(400, str(exc)) from exc


@router.post(
    "/prompts/{version_id}/promote",
    response_model=PromptPromoteResponse,
    openapi_extra={"x-path-example": {"version_id": "v2.0.0"}},
)
def promote_prompt(
    version_id: str,
    pipeline: CorrectionPipeline = Depends(get_pipeline),
) -> Any:
    """Promote a prompt version to active status."""

    try:
        return pipeline.promote_prompt(version_id)
    except KeyError as exc:
        raise _api_error(400, str(exc)) from exc


@router.post(
    "/prompts/rollback",
    response_model=PromptRollbackResponse,
    openapi_extra={"x-example-response": {"rolled_back_to": "v1.1.0"}},
)
def rollback_prompt(pipeline: CorrectionPipeline = Depends(get_pipeline)) -> Any:
    """Rollback to the previously active prompt version."""

    return pipeline.rollback_prompt()


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    openapi_extra=EVALUATE_OPENAPI,
)
def evaluate(
    payload: EvaluateRequest,
    pipeline: CorrectionPipeline = Depends(get_pipeline),
) -> Any:
    """Evaluate predictions against references with selected metrics."""

    try:
        return pipeline.evaluate_metrics(
            predictions=payload.predictions,
            references=payload.references,
            metrics=payload.metrics,
        )
    except ValueError as exc:
        raise _api_error(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive server path
        LOGGER.exception("Evaluation failed.")
        raise _api_error(500, str(exc)) from exc


@router.post(
    "/benchmark",
    response_model=BenchmarkReportResponse,
    openapi_extra=BENCHMARK_OPENAPI,
)
def benchmark(
    payload: BenchmarkRequest,
    pipeline: CorrectionPipeline = Depends(get_pipeline),
) -> Any:
    """Benchmark correction quality and latency over test pairs."""

    benchmark_pairs = [
        (item.original, item.reference)
        for item in payload.test_pairs[: payload.max_samples]
    ]
    try:
        report = pipeline.benchmark(benchmark_pairs)
        return asdict(report)
    except ValueError as exc:
        raise _api_error(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive server path
        LOGGER.exception("Benchmark failed.")
        raise _api_error(500, str(exc)) from exc

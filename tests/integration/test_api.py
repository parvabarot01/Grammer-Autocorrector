"""Integration tests for the FastAPI grammar correction service."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routes import get_runtime
from src.models import ErrorSpan
from src.pipeline import (
    BiasResult,
    FullGuardrailReport,
    GuardrailResult,
    PromptVersion,
    RetrievedChunk,
    ToxicityResult,
)


class _ActivePrompt:
    """Minimal prompt manager helper for `/info` coverage."""

    def __init__(self, version_id: str) -> None:
        self.version_id = version_id

    def get_active_prompt(self) -> "_ActivePrompt":
        return self


class FakeRuntime:
    """Mock runtime used to isolate API behavior from model dependencies."""

    def __init__(self) -> None:
        self.models_loaded = True
        self.prompt_manager = _ActivePrompt("v1.1.0")
        self.prompt_versions = {
            "v1.0.0": PromptVersion(
                version_id="v1.0.0",
                template="Correct: {input}",
                description="baseline",
                created_at="2026-05-25T00:00:00+00:00",
                metrics={"gleu": 0.61},
                is_active=False,
            ),
            "v1.1.0": PromptVersion(
                version_id="v1.1.0",
                template="Context: {context}\nSentence: {input}",
                description="rag",
                created_at="2026-05-25T00:01:00+00:00",
                metrics={"gleu": 0.68},
                is_active=True,
            ),
        }

    def serialize(self, value: Any) -> Any:
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
        if isinstance(value, list):
            return [self.serialize(item) for item in value]
        if isinstance(value, dict):
            return {key: self.serialize(item) for key, item in value.items()}
        return value

    def _report(self, original: str, corrected: str) -> FullGuardrailReport:
        input_valid = GuardrailResult(
            passed=True,
            violations=[],
            sanitized_text=original,
            severity="none",
        )
        output_valid = GuardrailResult(
            passed=True,
            violations=[],
            sanitized_text=corrected,
            severity="none",
        )
        return FullGuardrailReport(
            input_valid=input_valid,
            toxicity=ToxicityResult(score=0.0, is_toxic=False, detected_terms=[]),
            bias=BiasResult(has_bias=False, bias_types=[], suggestions=[]),
            output_valid=output_valid,
            overall_passed=True,
            timestamp="2026-05-25T00:00:00+00:00",
        )

    def correct(
        self,
        text: str,
        mode: str = "auto",
        num_beams: int = 4,
        return_detected_errors: bool = False,
        prompt_version: str | None = None,
    ) -> Dict[str, Any]:
        if not text.strip():
            raise ValueError("Input text cannot be empty.")
        corrected = text.replace(" go ", " goes ").replace("everyday", "every day")
        errors = [
            ErrorSpan(
                start=4,
                end=6,
                token="go",
                confidence=0.9,
                error_type="GRAMMAR_ERROR",
            )
        ]
        return {
            "original": text,
            "corrected": corrected,
            "mode_used": mode,
            "errors_detected": errors if return_detected_errors else None,
            "guardrail_report": self._report(text, corrected),
            "processing_time_ms": 12.5,
            "prompt_version": prompt_version or "v1.1.0",
        }

    def correct_batch(
        self,
        texts: List[str],
        mode: str = "auto",
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "original": text,
                "corrected": text.replace(" go ", " goes "),
                "mode_used": mode,
                "status": "success",
                "error": None,
                "processing_time_ms": 7.5,
                "batch_size": batch_size,
            }
            for text in texts
        ]

    def detect(self, text: str) -> Dict[str, Any]:
        errors = [
            ErrorSpan(
                start=4,
                end=6,
                token="go",
                confidence=0.88,
                error_type="GRAMMAR_ERROR",
            )
        ]
        return {
            "has_errors": True,
            "errors": errors,
            "error_count": 1,
            "processing_time_ms": 3.2,
        }

    def add_knowledge_rules(self, rules: List[str]) -> Dict[str, int]:
        return {"added": len(rules), "total_rules": 30 + len(rules)}

    def search_knowledge(self, query: str, top_k: int) -> Dict[str, Any]:
        return {
            "query": query,
            "results": [
                RetrievedChunk(
                    text="A singular subject normally takes a singular verb.",
                    score=0.12,
                    source="document_0",
                    chunk_id=0,
                )
                for _ in range(top_k)
            ],
        }

    def list_prompt_versions(self) -> Dict[str, Any]:
        versions = list(self.prompt_versions.values())
        return {"versions": versions, "active_version": "v1.1.0"}

    def get_prompt_version(self, version_id: str) -> PromptVersion:
        return self.prompt_versions[version_id]

    def promote_prompt(self, version_id: str) -> Dict[str, str]:
        for prompt in self.prompt_versions.values():
            prompt.is_active = False
        self.prompt_versions[version_id].is_active = True
        self.prompt_manager = _ActivePrompt(version_id)
        return {"promoted": version_id, "previous": "v1.1.0"}

    def rollback_prompt(self) -> Dict[str, str]:
        self.prompt_manager = _ActivePrompt("v1.0.0")
        return {"rolled_back_to": "v1.0.0"}

    def evaluate_metrics(
        self,
        predictions: List[str],
        references: List[str],
        metrics: List[str],
    ) -> Dict[str, Any]:
        return {
            "metrics": {"exact_match": 1.0, "gleu": 0.75, "rouge": {"rougeL": 0.8}},
            "evaluated_pairs": len(predictions),
        }


@pytest.fixture()
def client() -> TestClient:
    """Provide a test client with runtime dependency overrides."""

    fake_runtime = FakeRuntime()
    app.dependency_overrides[get_runtime] = lambda: fake_runtime
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_correct_endpoint_valid_input(client: TestClient) -> None:
    response = client.post(
        "/correct",
        json={"text": "She go to school everyday.", "return_detected_errors": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["corrected"] != payload["original"]
    assert payload["errors_detected"][0]["token"] == "go"


def test_correct_endpoint_empty_text_returns_400(client: TestClient) -> None:
    response = client.post("/correct", json={"text": "   "})
    assert response.status_code == 400


def test_correct_endpoint_text_too_long_returns_422(client: TestClient) -> None:
    response = client.post("/correct", json={"text": "a" * 1001})
    assert response.status_code == 422


def test_batch_correct_returns_correct_count(client: TestClient) -> None:
    response = client.post(
        "/correct/batch",
        json={"texts": ["She go home.", "He go outside."], "mode": "auto"},
    )
    assert response.status_code == 200
    assert response.json()["total_processed"] == 2


def test_detect_endpoint_valid_input(client: TestClient) -> None:
    response = client.post("/detect", json={"text": "She go to school everyday."})
    assert response.status_code == 200
    assert response.json()["has_errors"] is True


def test_knowledge_search_returns_results(client: TestClient) -> None:
    response = client.get(
        "/knowledge/search", params={"query": "She go school", "top_k": 2}
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


def test_prompts_list_returns_versions(client: TestClient) -> None:
    response = client.get("/prompts")
    assert response.status_code == 200
    assert response.json()["active_version"] == "v1.1.0"


def test_promote_and_rollback_prompt(client: TestClient) -> None:
    promote = client.post("/prompts/v1.0.0/promote")
    rollback = client.post("/prompts/rollback")
    assert promote.status_code == 200
    assert rollback.status_code == 200
    assert rollback.json()["rolled_back_to"] == "v1.0.0"


def test_evaluate_endpoint_returns_metrics(client: TestClient) -> None:
    response = client.post(
        "/evaluate",
        json={
            "predictions": ["She goes to school every day."],
            "references": ["She goes to school every day."],
            "metrics": ["gleu", "rouge", "exact_match"],
        },
    )
    assert response.status_code == 200
    assert response.json()["evaluated_pairs"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

"""Integration tests for the FastAPI grammar correction service."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routes import get_pipeline
from src.models import ErrorSpan
from src.pipeline import (
    BenchmarkReport,
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


class FakePipeline:
    """Mock pipeline used to isolate API behavior from model dependencies."""

    def __init__(self) -> None:
        self.models_loaded = True
        self.prompt_manager = _ActivePrompt("v1.1.0")
        self.prompt_versions = {
            "v1.0.0": PromptVersion(
                version_id="v1.0.0",
                template="Correct: {input}",
                description="baseline",
                created_at="2026-05-26T00:00:00+00:00",
                metrics={"gleu": 0.61},
                is_active=False,
            ),
            "v1.1.0": PromptVersion(
                version_id="v1.1.0",
                template="Context: {context}\nSentence: {input}",
                description="rag",
                created_at="2026-05-26T00:01:00+00:00",
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

    def utcnow(self) -> str:
        return "2026-05-26T00:00:00+00:00"

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
            timestamp="2026-05-26T00:00:00+00:00",
        )

    def correct(
        self,
        text: str,
        mode: str = "auto",
        num_beams: int = 4,
        return_errors: bool = False,
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
            "errors_detected": errors if return_errors else None,
            "guardrail_report": self._report(text, corrected),
            "processing_time_ms": 12.5,
            "model_version": "fake-t5",
            "prompt_version": prompt_version or "v1.1.0",
            "status": "success",
            "error_message": None,
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
                "errors_detected": None,
                "guardrail_report": self._report(text, text.replace(" go ", " goes ")),
                "status": "success",
                "error_message": None,
                "processing_time_ms": 7.5,
                "model_version": "fake-t5",
                "prompt_version": "v1.1.0",
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
                    chunk_id=index,
                )
                for index in range(top_k)
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

    def benchmark(self, test_data: List[tuple[str, str]]) -> BenchmarkReport:
        return BenchmarkReport(
            gleu=0.73,
            rouge={"rouge1": 0.81, "rouge2": 0.7, "rougeL": 0.79},
            exact_match=0.66,
            avg_latency_ms=11.2,
            p95_latency_ms=15.5,
            failure_rate=0.0,
            total_samples=len(test_data),
            timestamp="2026-05-26T00:00:00+00:00",
        )


@pytest.fixture()
def client() -> TestClient:
    """Provide a test client with pipeline dependency overrides."""

    fake_pipeline = FakePipeline()
    app.dependency_overrides[get_pipeline] = lambda: fake_pipeline
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


def test_public_correct_returns_only_public_safe_fields(client: TestClient) -> None:
    response = client.post("/public/correct", json={"text": "  She go to school.  "})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "original_text",
        "corrected_text",
        "changes",
        "summary",
        "success",
    }
    assert payload["original_text"] == "She go to school."
    assert payload["corrected_text"] == "She goes to school."
    assert payload["changes"][0]["before"] == "go"
    assert payload["changes"][0]["after"] == "goes"
    assert payload["success"] is True

    serialized_payload = json.dumps(payload).casefold()
    forbidden_terms = {
        "model",
        "model_name",
        "checkpoint",
        "prompt_version",
        "rag",
        "training",
        "evaluation",
        "debug",
        "fallback",
        "dataset",
        "guardrail",
        "t5",
        "bert",
        "rnn",
    }
    assert all(term not in serialized_payload for term in forbidden_terms)


def test_public_correct_empty_text_returns_clean_validation_error(
    client: TestClient,
) -> None:
    response = client.post("/public/correct", json={"text": "   "})

    assert response.status_code == 422
    assert "Input text cannot be empty." in response.text


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
        "/knowledge/search",
        params={"query": "She go school", "top_k": 2},
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


def test_knowledge_add_endpoint_returns_counts(client: TestClient) -> None:
    response = client.post(
        "/knowledge/add",
        json={"rules": ["Use 'an' before a vowel sound."]},
    )
    assert response.status_code == 200
    assert response.json()["added"] == 1


def test_benchmark_endpoint_returns_valid_report(client: TestClient) -> None:
    response = client.post(
        "/benchmark",
        json={
            "test_pairs": [
                {
                    "original": "She go to school everyday.",
                    "reference": "She goes to school every day.",
                }
            ],
            "max_samples": 1,
        },
    )
    assert response.status_code == 200
    assert response.json()["total_samples"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

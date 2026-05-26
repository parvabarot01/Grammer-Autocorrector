"""End-to-end tests covering the live API with a real correction pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routes import get_pipeline
from src.pipeline import CorrectionPipeline
from src.utils.config import (
    APIConfig,
    Config,
    DataConfig,
    GuardrailsConfig,
    ModelConfig,
    RAGConfig,
)


@pytest.fixture()
def real_pipeline(tmp_path: Path) -> CorrectionPipeline:
    """Create a real pipeline using test-local storage paths."""

    sample_data_path = Path(__file__).resolve().parents[2] / "data" / "sample"
    config = Config(
        model=ModelConfig(),
        data=DataConfig(
            raw_data_path=tmp_path / "raw",
            processed_data_path=tmp_path / "processed",
            sample_data_path=sample_data_path,
        ),
        rag=RAGConfig(vector_store_path=tmp_path / "vector_store"),
        api=APIConfig(),
        guardrails=GuardrailsConfig(),
    )
    pipeline = CorrectionPipeline(config)
    pipeline.load_all()
    return pipeline


@pytest.fixture()
def client(real_pipeline: CorrectionPipeline) -> TestClient:
    """Provide a test client backed by the real pipeline."""

    app.dependency_overrides[get_pipeline] = lambda: real_pipeline
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_full_pipeline_from_api_to_output(client: TestClient) -> None:
    response = client.post(
        "/correct",
        json={"text": "She go to school everyday.", "mode": "auto"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["corrected"] != payload["original"]
    assert payload["guardrail_report"]["overall_passed"] is True


def test_rag_mode_uses_retrieved_context(
    client: TestClient,
    real_pipeline: CorrectionPipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}
    original_augment_prompt = real_pipeline.rag.augment_prompt

    def tracking_augment_prompt(*args, **kwargs):
        calls["count"] += 1
        return original_augment_prompt(*args, **kwargs)

    monkeypatch.setattr(real_pipeline.rag, "augment_prompt", tracking_augment_prompt)

    response = client.post(
        "/correct",
        json={"text": "He have a apple.", "mode": "rag", "prompt_version": "v1.1.0"},
    )

    assert response.status_code == 200
    assert calls["count"] >= 1


def test_batch_endpoint_processes_all_items(client: TestClient) -> None:
    response = client.post(
        "/correct/batch",
        json={
            "texts": [
                "She go to school everyday.",
                "He have a apple.",
                "They goes there yesterday.",
            ],
            "mode": "auto",
            "batch_size": 2,
        },
    )

    assert response.status_code == 200
    assert response.json()["total_processed"] == 3


def test_benchmark_endpoint_returns_valid_report(client: TestClient) -> None:
    response = client.post(
        "/benchmark",
        json={
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
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_samples"] == 2
    assert 0.0 <= payload["gleu"] <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

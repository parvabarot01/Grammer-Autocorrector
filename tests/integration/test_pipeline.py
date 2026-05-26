"""Integration tests for prompt, guardrail, and RAG pipeline components."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.guardrails import GrammarGuardrails
from src.pipeline.prompt_versioning import PromptVersionManager
from src.pipeline.rag_pipeline import GrammarRAGPipeline


def test_rag_pipeline_build_and_retrieve(tmp_path: Path) -> None:
    pipeline = GrammarRAGPipeline(
        vector_store_path=str(tmp_path / "vector_store"),
        top_k=3,
    )
    rules = [
        "Use a singular verb with a singular subject.",
        "Use an article before a singular count noun.",
        "Keep tense consistent within the same time frame.",
        "Use commas to separate items in a series.",
        "Choose the correct preposition for time expressions.",
    ]

    pipeline.build_knowledge_base(rules)
    results = pipeline.retrieve("She go to school every day.", top_k=3)

    assert len(results) == 3
    assert all(result.text for result in results)


def test_prompt_versioning_promote_and_rollback(tmp_path: Path) -> None:
    manager = PromptVersionManager(str(tmp_path / "prompt_registry.json"))
    v1 = manager.register_prompt(
        template="Correct: {input}",
        description="custom baseline",
        version_id="v3.0.0",
    )
    v2 = manager.register_prompt(
        template="Context: {context}\nSentence: {input}",
        description="custom rag",
        version_id="v3.1.0",
    )

    manager.promote_prompt(v1.version_id)
    manager.promote_prompt(v2.version_id)
    rolled_back = manager.rollback()

    assert manager.get_active_prompt().version_id == v1.version_id
    assert rolled_back.version_id == v1.version_id


def test_full_pipeline_input_to_output(tmp_path: Path) -> None:
    pipeline = GrammarRAGPipeline(
        vector_store_path=str(tmp_path / "vector_store"),
        top_k=2,
    )
    pipeline.build_knowledge_base(
        [
            "A singular subject normally takes a singular verb.",
            "Use the article 'an' before a vowel sound.",
            "Use simple past for completed actions in the past.",
        ]
    )
    guardrails = GrammarGuardrails()

    input_result = guardrails.validate_input("She go to school everyday.")
    assert input_result.passed is True

    augmented_prompt = pipeline.augment_prompt(input_result.sanitized_text)

    assert "Relevant grammar guidance" in augmented_prompt
    assert "singular verb" in augmented_prompt or "article" in augmented_prompt


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

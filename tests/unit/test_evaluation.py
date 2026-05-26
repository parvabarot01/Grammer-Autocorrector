"""Unit tests for evaluation utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.utils.evaluation import Evaluator


def test_compute_gleu_with_fake_sacrebleu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = ModuleType("sacrebleu")
    fake_module.corpus_gleu = lambda predictions, references: SimpleNamespace(
        score=72.0
    )
    monkeypatch.setitem(sys.modules, "sacrebleu", fake_module)

    score = Evaluator().compute_gleu(
        ["She goes to school every day."],
        [["She goes to school every day."]],
    )

    assert score == 0.72


def test_compute_rouge_with_fake_rouge_score(monkeypatch: pytest.MonkeyPatch) -> None:
    rouge_score_module = ModuleType("rouge_score")
    rouge_scorer_module = ModuleType("rouge_score.rouge_scorer")

    class FakeScorer:
        def __init__(self, metrics, use_stemmer=True) -> None:
            self.metrics = metrics

        def score(self, reference, prediction):
            return {
                "rouge1": SimpleNamespace(fmeasure=0.8),
                "rouge2": SimpleNamespace(fmeasure=0.7),
                "rougeL": SimpleNamespace(fmeasure=0.75),
            }

    rouge_scorer_module.RougeScorer = FakeScorer
    rouge_score_module.rouge_scorer = rouge_scorer_module
    monkeypatch.setitem(sys.modules, "rouge_score", rouge_score_module)
    monkeypatch.setitem(sys.modules, "rouge_score.rouge_scorer", rouge_scorer_module)

    metrics = Evaluator().compute_rouge(
        ["She goes to school every day."],
        ["She goes to school every day."],
    )

    assert metrics["rouge1"] == 0.8
    assert metrics["rougeL"] == 0.75


def test_compute_exact_match_is_case_insensitive() -> None:
    score = Evaluator().compute_exact_match(
        ["She Goes To School Every Day."],
        ["she goes to school every day."],
    )
    assert score == 1.0


def test_compute_gleu_empty_predictions_returns_zero() -> None:
    assert Evaluator().compute_gleu([], []) == 0.0


def test_compute_rouge_empty_predictions_returns_zero_dict() -> None:
    metrics = Evaluator().compute_rouge([], [])
    assert metrics == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def test_compute_m2_score_returns_precision_recall_and_f05() -> None:
    metrics = Evaluator().compute_m2_score(
        original=["She go school."],
        corrected=["She goes school."],
        gold_edits=[["replace:1:2:goes"]],
    )

    assert set(metrics) == {"precision", "recall", "f05"}
    assert 0.0 <= metrics["precision"] <= 1.0


def test_evaluate_batch_returns_all_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = Evaluator()
    monkeypatch.setattr(evaluator, "compute_gleu", lambda predictions, references: 0.71)
    monkeypatch.setattr(
        evaluator,
        "compute_rouge",
        lambda predictions, references: {"rouge1": 0.8, "rouge2": 0.7, "rougeL": 0.75},
    )
    monkeypatch.setattr(
        evaluator, "compute_exact_match", lambda predictions, references: 0.5
    )
    monkeypatch.setattr(
        evaluator,
        "compute_m2_score",
        lambda original, corrected, gold_edits: {
            "precision": 0.6,
            "recall": 0.55,
            "f05": 0.58,
        },
    )

    dataset = [
        {
            "original": "She go school.",
            "references": ["She goes to school."],
            "gold_edits": ["replace:1:2:goes"],
        }
    ]
    metrics = evaluator.evaluate_batch(
        lambda batch: ["She goes to school."], dataset, batch_size=1
    )

    assert metrics["samples_evaluated"] == 1
    assert metrics["gleu"] == 0.71
    assert metrics["m2"]["f05"] == 0.58


def test_generate_evaluation_report_writes_markdown(tmp_path: Path) -> None:
    metrics = {
        "model_name": "demo-corrector",
        "gleu": 0.72,
        "rouge": {"rouge1": 0.8, "rouge2": 0.7, "rougeL": 0.75},
        "exact_match": 0.6,
    }
    output_path = tmp_path / "evaluation_report.md"

    Evaluator().generate_evaluation_report(metrics, str(output_path))

    contents = output_path.read_text(encoding="utf-8")
    assert "# Evaluation Report" in contents
    assert "demo-corrector" in contents


def test_extract_edits_and_validation_helpers() -> None:
    evaluator = Evaluator()
    edits = evaluator._extract_edits("She go school.", "She goes to school.")
    normalized = evaluator._normalize_edit("A replace:1:2:goes")

    assert edits
    assert normalized == "replace:1:2:goes"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

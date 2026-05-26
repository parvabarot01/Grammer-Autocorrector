"""Unit tests for the BERT grammar detector wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.bert_detector import BERTGrammarDetector, ErrorSpan


class DummyTokenizer:
    """Tokenizer stub for BERT tests."""

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return {
                "input_ids": [[101, 201, 202, 102]],
                "attention_mask": [[1, 1, 1, 1]],
                "token_type_ids": [[0, 0, 0, 0]],
                "offset_mapping": [[[0, 0], [0, 2], [3, 8], [0, 0]]],
            }
        return {
            "input_ids": [[101, 201, 202, 102] for _ in texts],
            "attention_mask": [[1, 1, 1, 1] for _ in texts],
            "token_type_ids": [[0, 0, 0, 0] for _ in texts],
            "offset_mapping": [[[0, 0], [0, 2], [3, 8], [0, 0]] for _ in texts],
        }

    def convert_ids_to_tokens(self, token_ids):
        return ["[CLS]", "He", "goes", "[SEP]"]

    def save_pretrained(self, output_dir):
        Path(output_dir, "bert-tokenizer.mock").write_text(
            "tokenizer",
            encoding="utf-8",
        )


class DummyModel:
    """Model stub for BERT tests."""

    def __init__(self, logits):
        self._logits = logits

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        return type("Output", (), {"logits": self._logits})()

    def parameters(self):
        return []

    def save_pretrained(self, output_dir):
        Path(output_dir, "bert-model.mock").write_text("model", encoding="utf-8")


def build_detector(logits) -> BERTGrammarDetector:
    """Create a detector seeded with dummy components."""

    detector = BERTGrammarDetector()
    detector.tokenizer = DummyTokenizer()
    detector.model = DummyModel(logits)
    return detector


def test_detect_errors_returns_list() -> None:
    detector = build_detector([[[3.0, 0.1], [3.0, 0.1], [0.1, 3.0], [2.0, 0.1]]])
    errors = detector.detect_errors("He goes")

    assert isinstance(errors, list)
    assert len(errors) == 1
    assert errors[0].token == "goes"


def test_has_errors_correct_sentence_returns_false() -> None:
    detector = build_detector([[[3.0, 0.1], [3.0, 0.1], [3.0, 0.1], [2.0, 0.1]]])
    assert detector.has_errors("He goes") is False


def test_error_span_dataclass_fields() -> None:
    span = ErrorSpan(start=1, end=4, token="bad", confidence=0.9, error_type="GRAMMAR")
    assert span.start == 1
    assert span.end == 4
    assert span.token == "bad"
    assert span.confidence == 0.9
    assert span.error_type == "GRAMMAR"


def test_detect_batch_length_matches() -> None:
    detector = build_detector(
        [
            [[3.0, 0.1], [0.1, 0.2], [0.1, 3.0], [2.0, 0.1]],
            [[3.0, 0.1], [3.0, 0.1], [0.1, 3.0], [2.0, 0.1]],
        ]
    )
    results = detector.detect_batch(["He goes", "She goes"], batch_size=2)

    assert len(results) == 2
    assert all(isinstance(item, list) for item in results)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

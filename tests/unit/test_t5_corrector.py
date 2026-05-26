"""Unit tests for the T5 grammar corrector wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.t5_corrector import T5GrammarCorrector


class DummyTokenizer:
    """Tokenizer stub for T5 tests."""

    def __init__(self) -> None:
        self.calls = []
        self.pad_token_id = 0

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {
            "input_ids": [[1, 2, 3] for _ in texts],
            "attention_mask": [[1, 1, 1] for _ in texts],
        }

    def decode(self, sequence, skip_special_tokens=True):
        return "decoded sentence"

    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["decoded sentence" for _ in sequences]

    def save_pretrained(self, output_dir):
        Path(output_dir, "tokenizer.mock").write_text("tokenizer", encoding="utf-8")


class DummyModel:
    """Model stub for T5 tests."""

    def __init__(self) -> None:
        self.saved_to = None

    def to(self, device):
        self.device = device
        return self

    def generate(self, **kwargs):
        batch_size = len(kwargs["input_ids"])
        return [[10, 11, 12] for _ in range(batch_size)]

    def parameters(self):
        return []

    def save_pretrained(self, output_dir):
        Path(output_dir, "model.mock").write_text("model", encoding="utf-8")


@pytest.fixture()
def corrector() -> T5GrammarCorrector:
    """Return a reusable corrector instance with test-safe defaults."""

    instance = T5GrammarCorrector()
    instance.tokenizer = DummyTokenizer()
    instance.model = DummyModel()
    return instance


def test_preprocess_adds_prefix(corrector: T5GrammarCorrector) -> None:
    batch = corrector.preprocess(["She go to school."])
    tokenizer = corrector.tokenizer

    assert tokenizer.calls[0][0] == ["grammar: She go to school."]
    assert "input_ids" in batch


def test_correct_returns_string(corrector: T5GrammarCorrector) -> None:
    result = corrector.correct("She go to school.")
    assert isinstance(result, str)
    assert result == "decoded sentence"


def test_correct_batch_length_matches_input(corrector: T5GrammarCorrector) -> None:
    results = corrector.correct_batch(
        ["One sentence.", "Two sentence."],
        batch_size=1,
    )
    assert len(results) == 2


def test_save_creates_directory(corrector: T5GrammarCorrector, tmp_path: Path) -> None:
    output_dir = tmp_path / "t5-output"
    corrector.save(str(output_dir))

    assert output_dir.exists()
    assert (output_dir / "model.mock").exists()
    assert (output_dir / "tokenizer.mock").exists()
    assert (output_dir / "t5_corrector_config.json").exists()


def test_from_pretrained_loads_correctly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()

    class DummyModelLoader:
        @staticmethod
        def from_pretrained(model_dir):
            return dummy_model

    class DummyTokenizerLoader:
        @staticmethod
        def from_pretrained(model_dir):
            return dummy_tokenizer

    def fake_import_transformers(self):
        return {
            "DataCollatorForSeq2Seq": object,
            "T5ForConditionalGeneration": DummyModelLoader,
            "T5Tokenizer": DummyTokenizerLoader,
            "Trainer": object,
            "TrainingArguments": object,
        }

    monkeypatch.setattr(
        T5GrammarCorrector,
        "_import_transformers",
        fake_import_transformers,
    )
    monkeypatch.setattr(T5GrammarCorrector, "_move_model_to_device", lambda self: None)

    (tmp_path / "t5_corrector_config.json").write_text(
        '{"max_length": 64, "batch_size": 8, "learning_rate": 0.001, "num_beams": 2}',
        encoding="utf-8",
    )

    instance = T5GrammarCorrector.from_pretrained(str(tmp_path))

    assert instance.model is dummy_model
    assert instance.tokenizer is dummy_tokenizer
    assert instance.max_length == 64
    assert instance.batch_size == 8


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

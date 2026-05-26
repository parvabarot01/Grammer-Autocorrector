"""Unit tests for the GrammarPreprocessor utility class."""

from __future__ import annotations

import pytest

from src.utils.preprocessing import GrammarPreprocessor


class DummyTokenizer:
    """Tokenizer stub that records inputs and returns deterministic payloads."""

    def __init__(self, include_token_type_ids: bool = False) -> None:
        self.include_token_type_ids = include_token_type_ids
        self.calls = []

    def __call__(self, text: str, **kwargs):
        self.calls.append({"text": text, "kwargs": kwargs})
        payload = {"input_ids": [101, 102], "attention_mask": [1, 1]}
        if self.include_token_type_ids:
            payload["token_type_ids"] = [0, 0]
        return payload


@pytest.fixture()
def preprocessor() -> GrammarPreprocessor:
    """Return a reusable preprocessor instance."""

    return GrammarPreprocessor()


@pytest.fixture()
def sample_batch() -> list[str]:
    """Return reusable batch input examples."""

    return ["  First sentence.  ", "", "Second\u200b sentence", "   "]


def test_clean_text_strips_whitespace(preprocessor: GrammarPreprocessor) -> None:
    assert preprocessor.clean_text("  Hello world.  ") == "Hello world."


def test_clean_text_normalizes_unicode(preprocessor: GrammarPreprocessor) -> None:
    assert preprocessor.clean_text("Cafe\u0301") == "Caf\u00e9"


def test_clean_text_collapses_spaces(preprocessor: GrammarPreprocessor) -> None:
    assert preprocessor.clean_text("Too   many\tspaces") == "Too many spaces"


def test_clean_text_removes_zero_width_characters(
    preprocessor: GrammarPreprocessor,
) -> None:
    assert preprocessor.clean_text("zero\u200bwidth") == "zerowidth"


def test_split_sentences_basic(preprocessor: GrammarPreprocessor) -> None:
    text = "One sentence. Two sentence? Three sentence!"
    assert preprocessor.split_into_sentences(text) == [
        "One sentence.",
        "Two sentence?",
        "Three sentence!",
    ]


def test_split_sentences_handles_abbreviations(
    preprocessor: GrammarPreprocessor,
) -> None:
    text = "Dr. Watson arrived late. He apologized immediately."
    assert preprocessor.split_into_sentences(text) == [
        "Dr. Watson arrived late.",
        "He apologized immediately.",
    ]


def test_split_sentences_empty_text_returns_empty_list(
    preprocessor: GrammarPreprocessor,
) -> None:
    assert preprocessor.split_into_sentences("   ") == []


def test_validate_input_empty_string(preprocessor: GrammarPreprocessor) -> None:
    is_valid, error_message = preprocessor.validate_input("   ")
    assert is_valid is False
    assert "cannot be empty" in error_message


def test_validate_input_exceeds_max_length(
    preprocessor: GrammarPreprocessor,
) -> None:
    is_valid, error_message = preprocessor.validate_input("abcdef", max_length=3)
    assert is_valid is False
    assert "maximum length" in error_message


def test_validate_input_non_printable_characters(
    preprocessor: GrammarPreprocessor,
) -> None:
    is_valid, error_message = preprocessor.validate_input("bad\x00text")
    assert is_valid is False
    assert "non-printable" in error_message


def test_validate_input_valid_text(preprocessor: GrammarPreprocessor) -> None:
    assert preprocessor.validate_input("Everything looks fine.") == (True, "")


def test_batch_preprocess_filters_empty(
    preprocessor: GrammarPreprocessor, sample_batch: list[str]
) -> None:
    assert preprocessor.batch_preprocess(sample_batch) == [
        "First sentence.",
        "Second sentence",
    ]


def test_tokenize_for_t5_adds_prefix(preprocessor: GrammarPreprocessor) -> None:
    tokenizer = DummyTokenizer()
    encoded = preprocessor.tokenize_for_t5(
        "Fix this sentence.", tokenizer, max_length=32
    )

    assert tokenizer.calls[0]["text"] == "grammar: Fix this sentence."
    assert "input_ids" in encoded
    assert "attention_mask" in encoded


def test_tokenize_for_bert_adds_token_type_ids_if_missing(
    preprocessor: GrammarPreprocessor,
) -> None:
    tokenizer = DummyTokenizer()
    encoded = preprocessor.tokenize_for_bert(
        "Classify this text.", tokenizer, max_length=32
    )

    assert encoded["token_type_ids"] == [0, 0]


def test_tokenize_for_bert_preserves_existing_token_type_ids(
    preprocessor: GrammarPreprocessor,
) -> None:
    tokenizer = DummyTokenizer(include_token_type_ids=True)
    encoded = preprocessor.tokenize_for_bert(
        "Keep provided ids.", tokenizer, max_length=32
    )

    assert encoded["token_type_ids"] == [0, 0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

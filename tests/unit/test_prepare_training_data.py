"""Unit tests for Sprint 8 training-data preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_training_data import (
    clean_dataset,
    generate_token_labels,
    normalize_dataset_columns,
    normalize_error_type,
    oversample_training_data,
    prepare_training_data,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


@pytest.fixture()
def sample_workspace(tmp_path: Path) -> dict[str, Path]:
    """Create a temporary raw/processed dataset layout for preparation tests."""

    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    output_dir = tmp_path / "data" / "processed" / "clean"
    results_dir = tmp_path / "results" / "data_preparation"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = [
        {
            "Serial Number": 1,
            "Error Type": "Subject-Verb Agreement",
            "Ungrammatical Statement": "He go home.",
            "Standard English": "He goes home.",
        },
        {
            "Serial Number": 2,
            "Error Type": "Article Usage",
            "Ungrammatical Statement": "She bought apple.",
            "Standard English": "She bought an apple.",
        },
        {
            "Serial Number": 3,
            "Error Type": "Subject-Verb Agreement",
            "Ungrammatical Statement": "He go home.",
            "Standard English": "He goes home.",
        },
        {
            "Serial Number": 4,
            "Error Type": "Capitalization Errors",
            "Ungrammatical Statement": "hello world.",
            "Standard English": "Hello world.",
        },
        {
            "Serial Number": 5,
            "Error Type": "Pronoun Errors",
            "Ungrammatical Statement": "The dog chased it's tail.",
            "Standard English": "The dog chased its tail.",
        },
        {
            "Serial Number": 6,
            "Error Type": "Ambiguity",
            "Ungrammatical Statement": "The dog chased it's tail.",
            "Standard English": "The dog chased its tail.",
        },
        {
            "Serial Number": 7,
            "Error Type": "Sentence Structure Errors",
            "Ungrammatical Statement": "Already correct.",
            "Standard English": "Already correct.",
        },
        {
            "Serial Number": 8,
            "Error Type": "No Error",
            "Ungrammatical Statement": "Clean sentence.",
            "Standard English": "Clean sentence.",
        },
        {
            "Serial Number": 9,
            "Error Type": "Subject-Verb Agreement",
            "Ungrammatical Statement": "They goes out.",
            "Standard English": "They go out.",
        },
    ]
    pd.DataFrame(raw_rows).to_csv(raw_dir / "grammar_correction.csv", index=False)

    def make_record(row: dict[str, object], split: str) -> dict[str, object]:
        return {
            "serial_number": row["Serial Number"],
            "error_type": row["Error Type"],
            "original": row["Ungrammatical Statement"],
            "corrected": row["Standard English"],
            "references": [row["Standard English"]],
            "source": "grammar_correction.csv",
            "split": split,
        }

    train_records = [
        make_record(raw_rows[0], "train"),
        make_record(raw_rows[1], "train"),
        make_record(raw_rows[2], "train"),
        make_record(raw_rows[5], "train"),
        make_record(raw_rows[8], "train"),
    ]
    validation_records = [
        make_record(raw_rows[3], "validation"),
        make_record(raw_rows[4], "validation"),
    ]
    test_records = [
        make_record(raw_rows[6], "test"),
        make_record(raw_rows[7], "test"),
    ]

    full_records = train_records + validation_records + test_records
    _write_jsonl(processed_dir / "grammar_correction_full.jsonl", full_records)
    _write_jsonl(processed_dir / "grammar_correction_train.jsonl", train_records)
    _write_jsonl(
        processed_dir / "grammar_correction_validation.jsonl", validation_records
    )
    _write_jsonl(processed_dir / "grammar_correction_test.jsonl", test_records)

    return {
        "input_path": raw_dir / "grammar_correction.csv",
        "processed_dir": processed_dir,
        "output_dir": output_dir,
        "results_dir": results_dir,
    }


def test_duplicate_removal(sample_workspace: dict[str, Path]) -> None:
    raw_frame = pd.read_csv(sample_workspace["input_path"])
    normalized = normalize_dataset_columns(raw_frame, "grammar_correction.csv")
    normalized["split"] = [
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "train",
        "test",
        "test",
        "train",
    ]

    clean, removed, flagged, counters = clean_dataset(normalized)

    assert len(clean) == 6
    assert counters["exact_duplicates_removed"] == 1
    assert counters["leakage_rows_removed"] == 1
    assert "exact_duplicate_pair" in removed["issue_reason"].tolist()
    assert any(
        reason.startswith("split_leakage_duplicate_kept_in_train")
        for reason in removed["issue_reason"].tolist()
    )
    assert len(flagged) == 1


def test_original_equals_corrected_is_flagged_unless_no_error() -> None:
    assert (
        normalize_error_type("No Error", "Clean sentence.", "Clean sentence.")
        == "no_error"
    )
    assert (
        normalize_error_type(
            "Sentence Structure Errors", "Already correct.", "Already correct."
        )
        == "mixed_multiple"
    )


def test_error_type_normalization() -> None:
    assert (
        normalize_error_type("Subject-Verb Agreement", "He go home.", "He goes home.")
        == "subject_verb_agreement"
    )
    assert (
        normalize_error_type(
            "Article Usage", "She bought apple.", "She bought an apple."
        )
        == "article"
    )
    assert (
        normalize_error_type("Capitalization Errors", "hello", "Hello")
        == "capitalization"
    )
    assert (
        normalize_error_type(
            "Pronoun Errors", "The dog chased it's tail.", "The dog chased its tail."
        )
        == "other"
    )


def test_oversampling_creates_more_balanced_classes(
    sample_workspace: dict[str, Path],
) -> None:
    raw_frame = pd.read_csv(sample_workspace["input_path"])
    normalized = normalize_dataset_columns(raw_frame, "grammar_correction.csv")
    normalized["split"] = [
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "train",
        "test",
        "test",
        "train",
    ]
    clean, _, _, _ = clean_dataset(normalized)
    train_clean = clean.loc[clean["split"] == "train"].copy()

    balanced, metadata = oversample_training_data(
        train_clean, seed=42, max_class_size=None, enable_oversampling=True
    )

    counts = balanced["normalized_error_type"].value_counts().to_dict()
    assert counts["subject_verb_agreement"] == 2
    assert counts["article"] == 2
    assert counts["mixed_multiple"] == 2
    assert metadata["oversample_counts"]["article"] == 1


def test_generate_token_labels_returns_aligned_tokens_and_labels() -> None:
    tokens, labels = generate_token_labels("He go home.", "He goes home.")

    assert len(tokens) == len(labels)
    assert tokens == ["He", "go", "home."]
    assert labels[1] == 1


def test_prepare_training_data_creates_output_files(
    sample_workspace: dict[str, Path],
) -> None:
    summary = prepare_training_data(
        input_path=sample_workspace["input_path"],
        processed_dir=sample_workspace["processed_dir"],
        output_dir=sample_workspace["output_dir"],
        results_dir=sample_workspace["results_dir"],
        seed=42,
        oversample=True,
    )

    assert summary["cleaned_full_count"] == 6
    assert summary["clean_train_count"] == 4
    assert summary["balanced_train_count"] == 6
    assert summary["validation_count"] == 1
    assert summary["test_count"] == 1

    expected_output_files = [
        "grammar_correction_clean_full.jsonl",
        "grammar_correction_clean_train.jsonl",
        "grammar_correction_clean_validation.jsonl",
        "grammar_correction_clean_test.jsonl",
        "grammar_correction_balanced_train.jsonl",
        "bert_token_labels_train.jsonl",
        "removed_rows.jsonl",
        "flagged_noisy_rows.jsonl",
    ]
    for filename in expected_output_files:
        assert (sample_workspace["output_dir"] / filename).exists()

    expected_result_files = [
        "sprint8_data_preparation_report.md",
        "class_distribution_before_after.csv",
        "bert_token_label_summary.json",
        "error_type_distribution_before.svg",
        "error_type_distribution_after_cleaning.svg",
        "error_type_distribution_after_balancing.svg",
        "sentence_length_distribution_cleaned.svg",
        "bert_token_label_distribution.svg",
    ]
    for filename in expected_result_files:
        assert (sample_workspace["results_dir"] / filename).exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

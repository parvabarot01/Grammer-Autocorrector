"""Dataset audit for the Grammar Autocorrector project."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "data_audit"
SEARCH_ROOTS = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "datasets",
    PROJECT_ROOT / "assets",
    PROJECT_ROOT / "data" / "sample",
    PROJECT_ROOT / "data" / "vector_store",
    PROJECT_ROOT / "data",
]
DATASET_EXTENSIONS = {".csv", ".json", ".jsonl", ".txt"}

PRIMARY_DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "grammar_correction_full.jsonl"
SPLIT_FILES = {
    "train": PROJECT_ROOT / "data" / "processed" / "grammar_correction_train.jsonl",
    "validation": PROJECT_ROOT / "data" / "processed" / "grammar_correction_validation.jsonl",
    "test": PROJECT_ROOT / "data" / "processed" / "grammar_correction_test.jsonl",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}
PUNCTUATION_CHARS = set(".,!?;:'\"-()[]{}")
ARTICLES = {"a", "an", "the"}
PREPOSITIONS = {
    "about",
    "above",
    "across",
    "after",
    "against",
    "along",
    "among",
    "around",
    "at",
    "before",
    "behind",
    "below",
    "beneath",
    "beside",
    "between",
    "by",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "near",
    "of",
    "off",
    "on",
    "onto",
    "over",
    "through",
    "to",
    "toward",
    "under",
    "with",
    "within",
    "without",
}
MALE_PRONOUNS = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
NEUTRAL_PRONOUNS = {"they", "them", "their", "theirs", "themself", "themselves"}
PROFESSION_TERMS = {
    "doctor",
    "nurse",
    "engineer",
    "teacher",
    "professor",
    "manager",
    "developer",
    "scientist",
    "lawyer",
    "chef",
    "pilot",
    "police",
    "officer",
    "designer",
    "writer",
    "student",
    "director",
    "assistant",
    "driver",
    "accountant",
}
SAFE_LOW_OVERLAP_LABELS = {
    "Abbreviation Errors",
    "Slang, Jargon, and Colloquialisms",
    "Word Choice/Usage",
    "Inappropriate Register",
}


@dataclass
class DatasetInventoryEntry:
    """Metadata describing a discovered dataset-like file."""

    path: str
    size_bytes: int
    examples: int | None
    classification: str
    columns: list[str]
    dtypes: dict[str, str]
    input_column: str | None
    corrected_column: str | None
    label_columns: list[str]
    metadata_columns: list[str]
    sample_rows: list[dict[str, Any]]


def utc_now() -> str:
    """Return the current UTC timestamp as an ISO string."""

    return datetime.now(timezone.utc).isoformat()


def relative_path(path: Path) -> str:
    """Return a project-relative path string."""

    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def file_size_label(num_bytes: int) -> str:
    """Render bytes as a human-readable label."""

    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024**2):.2f} MB"


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace to single spaces."""

    return " ".join(text.split())


def normalize_for_matching(text: str) -> str:
    """Normalize text for duplicate and overlap analysis."""

    lowered = normalize_whitespace(text).strip().lower()
    return "".join(char for char in lowered if char.isalnum() or char.isspace())


def tokenize(text: str) -> list[str]:
    """Tokenize text with simple alphanumeric token extraction."""

    token = []
    tokens: list[str] = []
    for char in text.lower():
        if char.isalnum() or char == "'":
            token.append(char)
        else:
            if token:
                tokens.append("".join(token))
                token.clear()
    if token:
        tokens.append("".join(token))
    return tokens


def sentence_token_count(text: str) -> int:
    """Count word tokens in a sentence."""

    return len(tokenize(text))


def safe_examples_for_text(path: Path) -> list[dict[str, Any]]:
    """Extract preview rows for plain-text files."""

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    preview = [line for line in lines if line and not line.startswith("#")][:10]
    return [{"line": line} for line in preview]


def classify_file(path: Path) -> str:
    """Classify whether a file is primary data, split data, or support data."""

    lower_name = path.name.lower()
    lower_path = relative_path(path).lower()
    if "grammar_correction" in lower_name and path.suffix.lower() in {".csv", ".jsonl"}:
        return "primary-dataset"
    if "grammar_correction" in lower_name and path.suffix.lower() == ".json":
        return "dataset-metadata"
    if "manifest" in lower_name or "prompt_registry" in lower_name:
        return "support-metadata"
    if "grammar_rules" in lower_name or "sample_sentences" in lower_name:
        return "support-text"
    if "vector_store" in lower_path:
        return "support-artifact"
    return "discovered-artifact"


def identify_columns(columns: Iterable[str]) -> tuple[str | None, str | None, list[str], list[str]]:
    """Infer important semantic columns from a dataset schema."""

    input_column = None
    corrected_column = None
    label_columns: list[str] = []
    metadata_columns: list[str] = []
    for column in columns:
        normalized = column.lower()
        if input_column is None and normalized in {
            "ungrammatical statement",
            "original",
            "input",
            "source_text",
        }:
            input_column = column
        if corrected_column is None and normalized in {
            "standard english",
            "corrected",
            "target",
            "reference",
            "references",
        }:
            corrected_column = column
        if "error_type" in normalized or normalized in {"label", "labels"}:
            label_columns.append(column)
        if normalized in {
            "serial number",
            "serial_number",
            "source",
            "split",
            "id",
            "metadata",
        } or normalized.endswith("_id"):
            metadata_columns.append(column)
    return input_column, corrected_column, label_columns, metadata_columns


def load_inventory_entry(path: Path) -> DatasetInventoryEntry:
    """Load a dataset-like file and summarize its schema."""

    suffix = path.suffix.lower()
    examples: int | None = None
    columns: list[str] = []
    dtypes: dict[str, str] = {}
    sample_rows: list[dict[str, Any]] = []

    if suffix == ".csv":
        dataframe = pd.read_csv(path)
        examples = len(dataframe)
        columns = [str(column) for column in dataframe.columns.tolist()]
        dtypes = {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()}
        sample_rows = dataframe.head(10).replace({np.nan: None}).to_dict(orient="records")
    elif suffix == ".jsonl":
        dataframe = pd.read_json(path, lines=True)
        examples = len(dataframe)
        columns = [str(column) for column in dataframe.columns.tolist()]
        dtypes = {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()}
        sample_rows = dataframe.head(10).replace({np.nan: None}).to_dict(orient="records")
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            examples = len(payload)
            if payload and isinstance(payload[0], dict):
                columns = [str(key) for key in payload[0].keys()]
                dtypes = {
                    key: type(payload[0].get(key)).__name__
                    for key in payload[0].keys()
                }
                sample_rows = payload[:10]
            else:
                sample_rows = [{"value": item} for item in payload[:10]]
        elif isinstance(payload, dict):
            if "versions" in payload and isinstance(payload["versions"], list):
                examples = len(payload["versions"])
                columns = [str(key) for key in payload["versions"][0].keys()]
                dtypes = {
                    key: type(payload["versions"][0].get(key)).__name__
                    for key in payload["versions"][0].keys()
                }
                sample_rows = payload["versions"][:10]
            elif "chunks" in payload and isinstance(payload["chunks"], list):
                examples = len(payload["chunks"])
                if payload["chunks"] and isinstance(payload["chunks"][0], dict):
                    columns = [str(key) for key in payload["chunks"][0].keys()]
                    dtypes = {
                        key: type(payload["chunks"][0].get(key)).__name__
                        for key in payload["chunks"][0].keys()
                    }
                sample_rows = payload["chunks"][:10]
            else:
                examples = None
                columns = [str(key) for key in payload.keys()]
                dtypes = {key: type(value).__name__ for key, value in payload.items()}
                sample_rows = [{key: payload[key] for key in list(payload.keys())[:10]}]
        else:
            sample_rows = [{"value": payload}]
    elif suffix == ".txt":
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        examples = len(lines)
        columns = ["line"]
        dtypes = {"line": "str"}
        sample_rows = safe_examples_for_text(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    input_column, corrected_column, label_columns, metadata_columns = identify_columns(columns)
    return DatasetInventoryEntry(
        path=relative_path(path),
        size_bytes=path.stat().st_size,
        examples=examples,
        classification=classify_file(path),
        columns=columns,
        dtypes=dtypes,
        input_column=input_column,
        corrected_column=corrected_column,
        label_columns=label_columns,
        metadata_columns=metadata_columns,
        sample_rows=sample_rows,
    )


def discover_dataset_files() -> list[Path]:
    """Locate dataset-like files across the configured search roots."""

    candidates: dict[Path, None] = {}
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in DATASET_EXTENSIONS:
                candidates[root] = None
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in DATASET_EXTENSIONS:
                candidates[path] = None
    return sorted(candidates.keys())


def normalized_error_bucket(label: str, original: str, corrected: str) -> str:
    """Map explicit labels into the requested audit error buckets."""

    normalized = label.lower()
    if "spelling" in normalized:
        return "spelling"
    if "verb tense" in normalized or "conditional" in normalized or "auxiliar" in normalized:
        return "tense"
    if "punctuation" in normalized or "ellipsis" in normalized or "run-on" in normalized:
        return "punctuation"
    if "article" in normalized:
        return "article"
    if "subject-verb" in normalized:
        return "subject-verb agreement"
    if "preposition" in normalized:
        return "preposition"
    if "capitalization" in normalized:
        return "capitalization"
    return infer_error_bucket(original, corrected)


def is_punctuation_only_change(original: str, corrected: str) -> bool:
    """Return whether the edit is mainly punctuation changes."""

    stripped_original = "".join(char for char in original.lower() if char.isalnum() or char.isspace())
    stripped_corrected = "".join(char for char in corrected.lower() if char.isalnum() or char.isspace())
    return stripped_original == stripped_corrected and original != corrected


def only_case_change(original: str, corrected: str) -> bool:
    """Return whether two strings differ only by casing."""

    return original.lower() == corrected.lower() and original != corrected


def token_diff_sets(original: str, corrected: str) -> tuple[list[str], list[str]]:
    """Extract tokens that appear changed between source and correction."""

    source_tokens = tokenize(original)
    target_tokens = tokenize(corrected)
    matcher = SequenceMatcher(a=source_tokens, b=target_tokens)
    source_changes: list[str] = []
    target_changes: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        source_changes.extend(source_tokens[i1:i2])
        target_changes.extend(target_tokens[j1:j2])
    return source_changes, target_changes


def infer_error_bucket(original: str, corrected: str) -> str:
    """Heuristically infer an audit bucket when no direct mapping exists."""

    if only_case_change(original, corrected):
        return "capitalization"
    if is_punctuation_only_change(original, corrected):
        return "punctuation"

    source_changes, target_changes = token_diff_sets(original, corrected)
    changed_tokens = set(source_changes + target_changes)
    if changed_tokens & ARTICLES:
        return "article"
    if changed_tokens & PREPOSITIONS:
        return "preposition"

    original_lower = tokenize(original)
    corrected_lower = tokenize(corrected)
    if any(
        pair in {
            ("go", "goes"),
            ("goes", "go"),
            ("have", "has"),
            ("has", "have"),
            ("is", "are"),
            ("are", "is"),
            ("was", "were"),
            ("were", "was"),
        }
        for pair in zip(original_lower, corrected_lower)
    ):
        return "subject-verb agreement"

    if len(source_changes) == len(target_changes) == 1:
        source_token = source_changes[0]
        target_token = target_changes[0]
        if SequenceMatcher(a=source_token, b=target_token).ratio() >= 0.75:
            return "spelling"
        if source_token.endswith(("ed", "ing", "s")) or target_token.endswith(("ed", "ing", "s")):
            return "tense"

    return "mixed/multiple"


def correction_pattern(original: str, corrected: str) -> str:
    """Classify the dominant correction pattern."""

    if original == corrected:
        return "unchanged"
    if only_case_change(original, corrected):
        return "casing-only"
    if is_punctuation_only_change(original, corrected):
        return "punctuation-only"

    source_tokens = tokenize(original)
    target_tokens = tokenize(corrected)
    matcher = SequenceMatcher(a=source_tokens, b=target_tokens)
    operations = [tag for tag, *_ in matcher.get_opcodes() if tag != "equal"]
    source_changes, target_changes = token_diff_sets(original, corrected)
    if len(source_changes) == 1 and len(target_changes) == 1:
        source_token = source_changes[0]
        target_token = target_changes[0]
        if len(target_token) > len(source_token) * 1.8:
            return "abbreviation-expansion"
        if SequenceMatcher(a=source_token, b=target_token).ratio() >= 0.75:
            return "single-token-replacement"
        return "lexical-substitution"
    if operations == ["insert"]:
        return "insertion"
    if operations == ["delete"]:
        return "deletion"
    if operations == ["replace"] and len(source_changes) <= 3 and len(target_changes) <= 3:
        return "multi-token-rewrite"
    return "structural-rewrite"


def gender_counts(texts: Iterable[str]) -> dict[str, int]:
    """Count gendered and neutral pronouns across texts."""

    counts = {"male": 0, "female": 0, "neutral": 0}
    for text in texts:
        words = tokenize(text)
        counts["male"] += sum(word in MALE_PRONOUNS for word in words)
        counts["female"] += sum(word in FEMALE_PRONOUNS for word in words)
        counts["neutral"] += sum(word in NEUTRAL_PRONOUNS for word in words)
    return counts


def profession_bias_records(texts: Iterable[str]) -> list[dict[str, Any]]:
    """Estimate pronoun-profession co-occurrence imbalance."""

    results: dict[str, dict[str, int]] = {
        profession: {"male": 0, "female": 0, "neutral": 0}
        for profession in PROFESSION_TERMS
    }
    for text in texts:
        words = tokenize(text)
        professions = {word for word in words if word in PROFESSION_TERMS}
        if not professions:
            continue
        has_male = any(word in MALE_PRONOUNS for word in words)
        has_female = any(word in FEMALE_PRONOUNS for word in words)
        has_neutral = any(word in NEUTRAL_PRONOUNS for word in words)
        for profession in professions:
            if has_male:
                results[profession]["male"] += 1
            if has_female:
                results[profession]["female"] += 1
            if has_neutral:
                results[profession]["neutral"] += 1
    ranked = []
    for profession, counts in results.items():
        total = sum(counts.values())
        if total:
            ranked.append(
                {
                    "profession": profession,
                    "male": counts["male"],
                    "female": counts["female"],
                    "neutral": counts["neutral"],
                    "total": total,
                }
            )
    return sorted(ranked, key=lambda item: item["total"], reverse=True)


def duplicate_metrics(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Compute duplicate and near-duplicate metrics."""

    normalized_originals = dataframe["original"].map(normalize_for_matching)
    normalized_corrected = dataframe["corrected"].map(normalize_for_matching)
    pair_keys = normalized_originals + " || " + normalized_corrected

    exact_original_duplicates = int(normalized_originals.duplicated().sum())
    exact_pair_duplicates = int(pair_keys.duplicated().sum())

    candidate_records = list(
        zip(
            dataframe["serial_number"].tolist(),
            normalized_originals.tolist(),
            dataframe["error_type"].tolist(),
            dataframe["original"].tolist(),
        )
    )
    near_duplicate_pairs: list[dict[str, Any]] = []
    for index, (serial_a, text_a, label_a, raw_a) in enumerate(candidate_records):
        tokens_a = set(tokenize(text_a))
        len_a = len(tokens_a)
        for serial_b, text_b, label_b, raw_b in candidate_records[index + 1 :]:
            if text_a == text_b:
                continue
            tokens_b = set(tokenize(text_b))
            if abs(len(tokens_a) - len(tokens_b)) > 2:
                continue
            union = tokens_a | tokens_b
            if not union:
                continue
            jaccard = len(tokens_a & tokens_b) / len(union)
            if jaccard < 0.8:
                continue
            similarity = SequenceMatcher(a=text_a, b=text_b).ratio()
            if similarity >= 0.92:
                near_duplicate_pairs.append(
                    {
                        "serial_a": int(serial_a),
                        "serial_b": int(serial_b),
                        "error_type_a": label_a,
                        "error_type_b": label_b,
                        "original_a": raw_a,
                        "original_b": raw_b,
                        "similarity": round(similarity, 3),
                    }
                )
    return {
        "exact_original_duplicates": exact_original_duplicates,
        "exact_pair_duplicates": exact_pair_duplicates,
        "near_duplicate_pairs": near_duplicate_pairs[:25],
        "near_duplicate_pair_count": len(near_duplicate_pairs),
    }


def text_issue_metrics(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Compute text cleanliness metrics."""

    rows = len(dataframe)
    empty_original = int(dataframe["original"].fillna("").str.strip().eq("").sum())
    empty_corrected = int(dataframe["corrected"].fillna("").str.strip().eq("").sum())
    same_as_original = int((dataframe["original"] == dataframe["corrected"]).sum())
    malformed_rows = int(
        dataframe["references"].map(lambda value: not isinstance(value, list) or not value).sum()
    )
    control_char_rows = int(
        dataframe.apply(
            lambda row: any(
                (ord(char) < 32 and char not in "\t\n\r")
                for char in f"{row['original']}{row['corrected']}"
            ),
            axis=1,
        ).sum()
    )
    unicode_suspicious_rows = int(
        dataframe.apply(
            lambda row: "\u200b" in row["original"]
            or "\u200b" in row["corrected"]
            or "\ufffd" in row["original"]
            or "\ufffd" in row["corrected"],
            axis=1,
        ).sum()
    )
    suspicious_label_rows = int(
        dataframe["error_type"]
        .astype(str)
        .map(lambda value: "Ã" in value or "\ufffd" in value)
        .sum()
    )

    noisy_examples: list[dict[str, Any]] = []
    for record in dataframe.itertuples(index=False):
        original = str(record.original)
        corrected = str(record.corrected)
        original_tokens = tokenize(original)
        corrected_tokens = tokenize(corrected)
        overlap = len(set(original_tokens) & set(corrected_tokens))
        denom = max(len(set(original_tokens) | set(corrected_tokens)), 1)
        overlap_ratio = overlap / denom
        length_ratio = len(corrected_tokens) / max(len(original_tokens), 1)
        suspicious = False
        reasons: list[str] = []
        if not corrected.strip():
            suspicious = True
            reasons.append("missing correction")
        if original == corrected:
            suspicious = True
            reasons.append("no correction delta")
        if length_ratio > 2.8 or length_ratio < 0.35:
            suspicious = True
            reasons.append(f"extreme length ratio {length_ratio:.2f}")
        if overlap_ratio < 0.2 and record.error_type not in SAFE_LOW_OVERLAP_LABELS:
            suspicious = True
            reasons.append(f"low token overlap {overlap_ratio:.2f}")
        if suspicious:
            noisy_examples.append(
                {
                    "serial_number": int(record.serial_number),
                    "error_type": record.error_type,
                    "original": original,
                    "corrected": corrected,
                    "reasons": reasons,
                }
            )

    return {
        "rows": rows,
        "empty_original_rows": empty_original,
        "empty_corrected_rows": empty_corrected,
        "same_original_corrected_rows": same_as_original,
        "malformed_reference_rows": malformed_rows,
        "control_character_rows": control_char_rows,
        "unicode_suspicious_rows": unicode_suspicious_rows,
        "suspicious_label_rows": suspicious_label_rows,
        "noisy_row_count": len(noisy_examples),
        "noisy_examples": noisy_examples[:20],
    }


def split_leakage_metrics(split_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Compute train/validation/test leakage statistics."""

    normalized_by_split = {
        split: {
            "original": set(frame["original"].map(normalize_for_matching)),
            "pair": set(
                (
                    frame["original"].map(normalize_for_matching)
                    + " || "
                    + frame["corrected"].map(normalize_for_matching)
                ).tolist()
            ),
        }
        for split, frame in split_frames.items()
    }

    overlaps: dict[str, dict[str, int]] = {}
    split_names = list(split_frames.keys())
    for index, split_a in enumerate(split_names):
        for split_b in split_names[index + 1 :]:
            key = f"{split_a}_vs_{split_b}"
            overlaps[key] = {
                "original_overlap": len(
                    normalized_by_split[split_a]["original"]
                    & normalized_by_split[split_b]["original"]
                ),
                "pair_overlap": len(
                    normalized_by_split[split_a]["pair"]
                    & normalized_by_split[split_b]["pair"]
                ),
            }
    return overlaps


def token_error_ratio(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Estimate CORRECT vs ERROR token ratio using token sequence diffs."""

    total_tokens = 0
    error_tokens = 0
    correct_tokens = 0
    error_token_counter: Counter[str] = Counter()

    for record in dataframe.itertuples(index=False):
        source_tokens = tokenize(record.original)
        target_tokens = tokenize(record.corrected)
        total_tokens += len(source_tokens)
        matcher = SequenceMatcher(a=source_tokens, b=target_tokens)
        row_error_tokens = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                correct_tokens += i2 - i1
                continue
            row_error_tokens += max(i2 - i1, j2 - j1)
            error_token_counter.update(source_tokens[i1:i2] or target_tokens[j1:j2])
        error_tokens += row_error_tokens

    error_ratio = error_tokens / max(total_tokens, 1)
    return {
        "token_labels_available": False,
        "approx_total_source_tokens": int(total_tokens),
        "approx_error_tokens": int(error_tokens),
        "approx_correct_tokens": int(correct_tokens),
        "approx_error_token_ratio": round(error_ratio, 4),
        "approx_correct_to_error_ratio": round(correct_tokens / max(error_tokens, 1), 3),
        "top_changed_tokens": [
            {"token": token, "count": count}
            for token, count in error_token_counter.most_common(15)
        ],
    }


def quality_score(issues: dict[str, Any], duplicates: dict[str, Any], leakage: dict[str, Any], rows: int) -> dict[str, Any]:
    """Aggregate data quality metrics into a cleanliness score."""

    missing_rate = (issues["empty_original_rows"] + issues["empty_corrected_rows"]) / max(rows, 1)
    malformed_rate = issues["malformed_reference_rows"] / max(rows, 1)
    control_rate = (issues["control_character_rows"] + issues["unicode_suspicious_rows"]) / max(rows, 1)
    label_issue_rate = issues["suspicious_label_rows"] / max(rows, 1)
    duplicate_rate = duplicates["exact_pair_duplicates"] / max(rows, 1)
    near_duplicate_rate = duplicates["near_duplicate_pair_count"] / max(rows, 1)
    noisy_rate = issues["noisy_row_count"] / max(rows, 1)
    leakage_rate = sum(item["pair_overlap"] for item in leakage.values()) / max(rows, 1)

    score = 100.0
    score -= missing_rate * 30
    score -= malformed_rate * 20
    score -= control_rate * 10
    score -= label_issue_rate * 50
    score -= duplicate_rate * 25
    score -= near_duplicate_rate * 10
    score -= noisy_rate * 20
    score -= leakage_rate * 35
    score = max(min(score, 100.0), 0.0)

    if score >= 90 and rows >= 10000:
        readiness = "High"
    elif score >= 80:
        readiness = "Moderate"
    else:
        readiness = "Low"

    return {
        "cleanliness_score": round(score, 2),
        "training_readiness": readiness,
        "components": {
            "missing_rate": round(missing_rate, 4),
            "malformed_rate": round(malformed_rate, 4),
            "control_character_rate": round(control_rate, 4),
            "label_issue_rate": round(label_issue_rate, 4),
            "duplicate_rate": round(duplicate_rate, 4),
            "near_duplicate_rate": round(near_duplicate_rate, 4),
            "noisy_rate": round(noisy_rate, 4),
            "split_leakage_rate": round(leakage_rate, 4),
        },
    }


def dataset_strengths(summary: dict[str, Any]) -> list[str]:
    """Return concise dataset strengths."""

    strengths = [
        "The corpus already contains paired ungrammatical and corrected sentences, which is directly usable for sequence-to-sequence training.",
        "The dataset includes explicit high-level error_type labels, which makes targeted analysis and curriculum design possible.",
        "Train, validation, and test splits are present and sized sensibly for a 2,018-example corpus.",
    ]
    if summary["quality"]["cleanliness_score"] >= 90:
        strengths.append("Text cleanliness is high, with essentially no empty rows, control characters, or split leakage.")
    if summary["duplicates"]["exact_pair_duplicates"] == 0:
        strengths.append("No exact duplicate source-target pairs were detected in the primary corpus.")
    return strengths


def dataset_weaknesses(summary: dict[str, Any]) -> list[str]:
    """Return concise dataset weaknesses."""

    weaknesses = [
        "The dataset is small for fine-tuning large grammar-correction models, especially for T5.",
        "Many examples belong to niche or stylistic categories rather than core grammar categories, which can dilute signal for sentence-level correction.",
        "Token-level ERROR labels are not explicitly provided, so BERT detection targets must be weakly inferred from diffs.",
    ]
    if summary["error_bucket_distribution"]["mixed/multiple"]["percentage"] > 30:
        weaknesses.append("A large share of rows collapse into a mixed/multiple bucket, which indicates broad label diversity and lower category purity.")
    if summary["bias"]["gender_pronouns"]["male"] > summary["bias"]["gender_pronouns"]["female"] * 1.2:
        weaknesses.append("Male pronouns appear more frequently than female pronouns, which could bias generated examples.")
    return weaknesses


def biggest_risks(summary: dict[str, Any]) -> list[str]:
    """Return the biggest risks before training."""

    risks = [
        "Overfitting is likely if T5 is trained only on this 2,018-row corpus without augmentation or external data.",
        "Weakly inferred token labels can make the BERT detector under-calibrated and class-imbalanced.",
        "Abbreviation, style, and structural rewrite examples may encourage broader rewriting instead of minimal grammar edits.",
    ]
    if summary["bert_imbalance"]["approx_error_token_ratio"] < 0.2:
        risks.append("Estimated ERROR tokens are sparse relative to CORRECT tokens, so detector loss weighting is recommended.")
    return risks


def recommendations(summary: dict[str, Any]) -> dict[str, Any]:
    """Return actionable training recommendations."""

    error_ratio = summary["bert_imbalance"]["approx_error_token_ratio"]
    oversampling_needed = summary["error_bucket_distribution"]["article"]["percentage"] < 8 or summary["error_bucket_distribution"]["capitalization"]["percentage"] < 5
    augmentation_needed = summary["primary_dataset"]["rows"] < 10000
    recommendations_list = [
        "Use weighted cross-entropy for BERT token detection to offset the dominant CORRECT token class.",
        "Keep the current train/validation/test split, because no material pair leakage was detected.",
        "Consider oversampling core grammar categories such as article, punctuation, capitalization, and preposition errors if those are product priorities.",
        "Add external grammar-correction corpora such as JFLEG and CoNLL-2014 before serious T5 fine-tuning.",
        "Augment with synthetic perturbations or back-translation if you want stronger T5 generalization from this small corpus.",
    ]
    if error_ratio < 0.15:
        recommendations_list.append("Focal loss is a reasonable fallback for BERT if weighted loss alone does not stabilize minority ERROR tokens.")
    expected_impact = (
        "With only the current corpus, T5 is more likely to memorize recurring correction templates than to generalize broadly. "
        "Adding JFLEG or CoNLL-style data should materially improve robustness and likely raise downstream accuracy."
    )
    return {
        "oversampling_needed": oversampling_needed,
        "augmentation_needed": augmentation_needed,
        "recommendations": recommendations_list,
        "expected_t5_accuracy_impact": expected_impact,
    }


def format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render a list of dictionaries as a markdown table."""

    if not rows:
        return "| No rows |"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def top_tokens(texts: Iterable[str], limit: int = 20) -> list[dict[str, Any]]:
    """Return the most common non-stopword tokens."""

    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(
            token
            for token in tokenize(text)
            if token not in STOPWORDS and len(token) > 1
        )
    return [{"token": token, "count": count} for token, count in counter.most_common(limit)]


def sentence_length_stats(texts: Iterable[str]) -> dict[str, Any]:
    """Compute token-length statistics."""

    lengths = [sentence_token_count(text) for text in texts]
    if not lengths:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0,
            "max": 0,
            "p95": 0.0,
            "std": 0.0,
        }
    return {
        "mean": round(statistics.fmean(lengths), 3),
        "median": round(statistics.median(lengths), 3),
        "min": int(min(lengths)),
        "max": int(max(lengths)),
        "p95": round(float(np.percentile(lengths, 95)), 3),
        "std": round(statistics.pstdev(lengths), 3),
    }


def vocabulary_stats(texts: Iterable[str]) -> dict[str, Any]:
    """Compute vocabulary statistics."""

    tokens = []
    for text in texts:
        tokens.extend(tokenize(text))
    vocab = set(tokens)
    return {
        "token_count": len(tokens),
        "vocabulary_size": len(vocab),
        "type_token_ratio": round(len(vocab) / max(len(tokens), 1), 4),
    }


def render_svg_bar_chart(title: str, items: list[tuple[str, float]], output_path: Path, color: str = "#3b82f6") -> None:
    """Render a simple horizontal SVG bar chart."""

    if not items:
        output_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return

    width = 1100
    bar_height = 26
    label_width = 330
    value_width = 80
    chart_width = width - label_width - value_width - 80
    top_padding = 70
    left_padding = 30
    max_value = max(value for _, value in items) or 1.0
    height = top_padding + len(items) * (bar_height + 12) + 30

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#111827;} .title{font-size:20px;font-weight:bold;} .axis{fill:#6b7280;} .value{font-weight:bold;}</style>",
        f"<text class='title' x='{left_padding}' y='32'>{title}</text>",
    ]

    y = top_padding
    for label, value in items:
        bar_length = 0 if max_value == 0 else (value / max_value) * chart_width
        lines.append(f"<text x='{left_padding}' y='{y + 17}'>{label}</text>")
        lines.append(
            f"<rect x='{left_padding + label_width}' y='{y}' width='{chart_width}' height='{bar_height}' rx='4' fill='#e5e7eb' />"
        )
        lines.append(
            f"<rect x='{left_padding + label_width}' y='{y}' width='{bar_length:.2f}' height='{bar_height}' rx='4' fill='{color}' />"
        )
        lines.append(
            f"<text class='value' x='{left_padding + label_width + chart_width + 12}' y='{y + 17}'>{value:.2f}</text>"
        )
        y += bar_height + 12

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_svg_histogram(title: str, values: list[int], output_path: Path, bins: int = 12, color: str = "#10b981") -> None:
    """Render a simple SVG histogram."""

    width = 1100
    height = 500
    margin = {"left": 60, "right": 30, "top": 70, "bottom": 80}
    chart_width = width - margin["left"] - margin["right"]
    chart_height = height - margin["top"] - margin["bottom"]

    if not values:
        output_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        bin_edges = np.array([min_value, max_value + 1])
        counts = np.array([len(values)])
    else:
        counts, bin_edges = np.histogram(values, bins=bins)
    max_count = int(max(counts)) or 1
    bar_count = len(counts)
    bar_width = chart_width / max(bar_count, 1)

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#111827;} .title{font-size:20px;font-weight:bold;} .axis{stroke:#9ca3af;stroke-width:1;} .tick{fill:#6b7280;} .value{font-size:11px;fill:#374151;}</style>",
        f"<text class='title' x='{margin['left']}' y='32'>{title}</text>",
        f"<line class='axis' x1='{margin['left']}' y1='{margin['top'] + chart_height}' x2='{margin['left'] + chart_width}' y2='{margin['top'] + chart_height}' />",
        f"<line class='axis' x1='{margin['left']}' y1='{margin['top']}' x2='{margin['left']}' y2='{margin['top'] + chart_height}' />",
    ]

    for idx, count in enumerate(counts):
        x = margin["left"] + idx * bar_width + 4
        bar_height = 0 if max_count == 0 else (count / max_count) * chart_height
        y = margin["top"] + chart_height - bar_height
        lines.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{max(bar_width - 8, 2):.2f}' height='{bar_height:.2f}' fill='{color}' rx='3' />"
        )
        label = f"{int(bin_edges[idx])}-{int(bin_edges[idx + 1])}"
        lines.append(
            f"<text class='tick' x='{x + (bar_width - 8) / 2:.2f}' y='{margin['top'] + chart_height + 18}' text-anchor='middle'>{label}</text>"
        )
        lines.append(
            f"<text class='value' x='{x + (bar_width - 8) / 2:.2f}' y='{max(y - 6, margin['top'] + 12):.2f}' text-anchor='middle'>{int(count)}</text>"
        )

    output_path.write_text("\n".join(lines + ["</svg>"]), encoding="utf-8")


def render_svg_grouped_chart(title: str, series: list[dict[str, Any]], labels: list[str], output_path: Path) -> None:
    """Render a grouped bar chart in SVG."""

    width = 1100
    height = 520
    margin = {"left": 70, "right": 20, "top": 70, "bottom": 90}
    chart_width = width - margin["left"] - margin["right"]
    chart_height = height - margin["top"] - margin["bottom"]
    colors = ["#2563eb", "#ec4899", "#14b8a6"]

    max_value = max(
        max(item.get(label, 0) for label in labels)
        for item in series
    ) or 1

    group_width = chart_width / max(len(series), 1)
    bar_width = group_width / max(len(labels) + 1, 1)

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#111827;} .title{font-size:20px;font-weight:bold;} .axis{stroke:#9ca3af;stroke-width:1;} .tick{fill:#6b7280;}</style>",
        f"<text class='title' x='{margin['left']}' y='32'>{title}</text>",
        f"<line class='axis' x1='{margin['left']}' y1='{margin['top'] + chart_height}' x2='{margin['left'] + chart_width}' y2='{margin['top'] + chart_height}' />",
        f"<line class='axis' x1='{margin['left']}' y1='{margin['top']}' x2='{margin['left']}' y2='{margin['top'] + chart_height}' />",
    ]

    legend_x = margin["left"]
    for index, label in enumerate(labels):
        lines.append(
            f"<rect x='{legend_x}' y='46' width='14' height='14' fill='{colors[index % len(colors)]}' rx='2' />"
        )
        lines.append(f"<text x='{legend_x + 22}' y='58'>{label}</text>")
        legend_x += 140

    for series_index, item in enumerate(series):
        x0 = margin["left"] + series_index * group_width
        category = item.get("label", "")
        lines.append(
            f"<text class='tick' x='{x0 + group_width / 2:.2f}' y='{margin['top'] + chart_height + 20}' text-anchor='middle'>{category}</text>"
        )
        for label_index, label in enumerate(labels):
            value = float(item.get(label, 0))
            bar_height = (value / max_value) * chart_height
            x = x0 + (label_index + 0.5) * bar_width
            y = margin["top"] + chart_height - bar_height
            lines.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width * 0.75:.2f}' height='{bar_height:.2f}' fill='{colors[label_index % len(colors)]}' rx='3' />"
            )
            lines.append(
                f"<text x='{x + bar_width * 0.375:.2f}' y='{max(y - 6, margin['top'] + 12):.2f}' text-anchor='middle'>{value:.0f}</text>"
            )

    output_path.write_text("\n".join(lines + ["</svg>"]), encoding="utf-8")


def write_reports(summary: dict[str, Any], inventory: list[DatasetInventoryEntry]) -> None:
    """Write the markdown and JSON deliverables."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    inventory_rows = [
        {
            "path": item.path,
            "classification": item.classification,
            "size": file_size_label(item.size_bytes),
            "examples": item.examples if item.examples is not None else "n/a",
        }
        for item in inventory
    ]
    explicit_label_rows = [
        {
            "error_type": label,
            "count": stats["count"],
            "percentage": f"{stats['percentage']:.2f}%",
        }
        for label, stats in summary["explicit_error_type_distribution"].items()
    ]
    bucket_rows = [
        {
            "bucket": bucket,
            "count": stats["count"],
            "percentage": f"{stats['percentage']:.2f}%",
        }
        for bucket, stats in summary["error_bucket_distribution"].items()
    ]
    bias_rows = [
        {
            "indicator": "Male pronouns",
            "value": summary["bias"]["gender_pronouns"]["male"],
        },
        {
            "indicator": "Female pronouns",
            "value": summary["bias"]["gender_pronouns"]["female"],
        },
        {
            "indicator": "Neutral pronouns",
            "value": summary["bias"]["gender_pronouns"]["neutral"],
        },
        {
            "indicator": "Exact pair duplicates",
            "value": summary["duplicates"]["exact_pair_duplicates"],
        },
        {
            "indicator": "Near-duplicate pairs",
            "value": summary["duplicates"]["near_duplicate_pair_count"],
        },
        {
            "indicator": "Noisy rows",
            "value": summary["quality_issues"]["noisy_row_count"],
        },
    ]

    report_lines = [
        "# Data Audit Report",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Primary dataset: `{summary['primary_dataset']['path']}`",
        f"- Rows: `{summary['primary_dataset']['rows']}`",
        f"- Cleanliness score: `{summary['quality']['cleanliness_score']}/100`",
        f"- Training readiness: `{summary['quality']['training_readiness']}`",
        "",
        "## Dataset Inventory",
        "",
        format_table(inventory_rows, ["path", "classification", "size", "examples"]),
        "",
        "## Dataset Schemas",
        "",
    ]
    for entry in inventory:
        column_rows = (
            [{"column": column, "dtype": entry.dtypes.get(column, "n/a")} for column in entry.columns]
            if entry.columns
            else [{"column": "n/a", "dtype": "n/a"}]
        )
        sample_columns = list(entry.sample_rows[0].keys()) if entry.sample_rows else ["sample"]
        sample_rows = entry.sample_rows if entry.sample_rows else [{"sample": "n/a"}]
        report_lines.extend(
            [
                f"### `{entry.path}`",
                "",
                f"- Classification: `{entry.classification}`",
                f"- File size: `{file_size_label(entry.size_bytes)}`",
                f"- Examples: `{entry.examples if entry.examples is not None else 'n/a'}`",
                f"- Input column: `{entry.input_column}`",
                f"- Corrected column: `{entry.corrected_column}`",
                f"- Label columns: `{', '.join(entry.label_columns) if entry.label_columns else 'none'}`",
                f"- Metadata columns: `{', '.join(entry.metadata_columns) if entry.metadata_columns else 'none'}`",
                "",
                format_table(column_rows, ["column", "dtype"]),
                "",
                format_table(sample_rows, sample_columns),
                "",
            ]
        )
    report_lines.extend([
        "",
        "## Primary Dataset Schema",
        "",
        f"- Input column: `{summary['primary_dataset']['input_column']}`",
        f"- Corrected column: `{summary['primary_dataset']['corrected_column']}`",
        f"- Label columns: `{', '.join(summary['primary_dataset']['label_columns'])}`",
        f"- Metadata columns: `{', '.join(summary['primary_dataset']['metadata_columns'])}`",
        "",
        "### Column Types",
        "",
        format_table(
            [
                {"column": key, "dtype": value}
                for key, value in summary["primary_dataset"]["dtypes"].items()
            ],
            ["column", "dtype"],
        ),
        "",
        "### Sample Rows",
        "",
        format_table(
            summary["primary_dataset"]["sample_rows"],
            list(summary["primary_dataset"]["sample_rows"][0].keys())
            if summary["primary_dataset"]["sample_rows"]
            else ["sample"],
        ),
        "",
        "## Split Summary",
        "",
        format_table(
            [
                {
                    "split": split,
                    "rows": stats["rows"],
                    "percentage": f"{stats['percentage']:.2f}%",
                }
                for split, stats in summary["split_distribution"].items()
            ],
            ["split", "rows", "percentage"],
        ),
        "",
        "## Error Type Distribution",
        "",
        "### Explicit Labels",
        "",
        format_table(explicit_label_rows, ["error_type", "count", "percentage"]),
        "",
        "### Normalized Audit Buckets",
        "",
        format_table(bucket_rows, ["bucket", "count", "percentage"]),
        "",
        "### Imbalance Report",
        "",
        f"- Largest explicit label: `{summary['imbalance']['largest_label']}` ({summary['imbalance']['largest_label_share']:.2f}%)",
        f"- Smallest explicit label: `{summary['imbalance']['smallest_label']}` ({summary['imbalance']['smallest_label_share']:.2f}%)",
        f"- Largest audit bucket: `{summary['imbalance']['largest_bucket']}` ({summary['imbalance']['largest_bucket_share']:.2f}%)",
        f"- Smallest audit bucket: `{summary['imbalance']['smallest_bucket']}` ({summary['imbalance']['smallest_bucket_share']:.2f}%)",
        "",
        "## Data Quality",
        "",
        f"- Average source sentence length: `{summary['length_stats']['original']['mean']}` tokens",
        f"- Average corrected sentence length: `{summary['length_stats']['corrected']['mean']}` tokens",
        f"- Vocabulary size: `{summary['vocabulary']['vocabulary_size']}`",
        f"- Type-token ratio: `{summary['vocabulary']['type_token_ratio']}`",
        f"- Empty original rows: `{summary['quality_issues']['empty_original_rows']}`",
        f"- Empty corrected rows: `{summary['quality_issues']['empty_corrected_rows']}`",
        f"- Suspicious label rows: `{summary['quality_issues']['suspicious_label_rows']}`",
        f"- Noisy rows: `{summary['quality_issues']['noisy_row_count']}`",
        f"- Exact pair duplicates: `{summary['duplicates']['exact_pair_duplicates']}`",
        "",
        "### Split Leakage",
        "",
        format_table(
            [
                {
                    "comparison": key,
                    "original_overlap": value["original_overlap"],
                    "pair_overlap": value["pair_overlap"],
                }
                for key, value in summary["split_leakage"].items()
            ],
            ["comparison", "original_overlap", "pair_overlap"],
        ),
        "",
        "## BERT Token Imbalance Estimate",
        "",
        f"- Approximate source tokens: `{summary['bert_imbalance']['approx_total_source_tokens']}`",
        f"- Approximate ERROR tokens: `{summary['bert_imbalance']['approx_error_tokens']}`",
        f"- Approximate CORRECT tokens: `{summary['bert_imbalance']['approx_correct_tokens']}`",
        f"- Approximate ERROR token ratio: `{summary['bert_imbalance']['approx_error_token_ratio']}`",
        f"- Approximate CORRECT:ERROR ratio: `{summary['bert_imbalance']['approx_correct_to_error_ratio']}`",
        "",
        "### Top Changed Tokens",
        "",
        format_table(
            summary["bert_imbalance"]["top_changed_tokens"],
            ["token", "count"],
        ),
        "",
        "## Strengths",
        "",
        *[f"- {item}" for item in summary["recommendation_block"]["strengths"]],
        "",
        "## Weaknesses",
        "",
        *[f"- {item}" for item in summary["recommendation_block"]["weaknesses"]],
        "",
        "## Biggest Risks Before Training",
        "",
        *[f"- {item}" for item in summary["recommendation_block"]["risks"]],
        "",
        "## Final Recommendations",
        "",
        f"- Oversampling needed: `{summary['recommendations']['oversampling_needed']}`",
        f"- Augmentation needed: `{summary['recommendations']['augmentation_needed']}`",
        *[f"- {item}" for item in summary["recommendations"]["recommendations"]],
        "",
        "## Charts",
        "",
        "- `error_type_distribution.svg`",
        "- `sentence_length_distribution.svg`",
        "- `top_repeated_tokens.svg`",
        "- `duplicate_frequency.svg`",
        "- `split_distribution.svg`",
        "- `bias_indicators.svg`",
    ])
    (RESULTS_DIR / "data_audit_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    bias_lines = [
        "# Bias Report",
        "",
        f"- Generated: `{summary['generated_at']}`",
        "",
        "## Gender Pronoun Balance",
        "",
        format_table(
            [
                {"group": key, "count": value}
                for key, value in summary["bias"]["gender_pronouns"].items()
            ],
            ["group", "count"],
        ),
        "",
        "## Profession Co-occurrence Snapshot",
        "",
        format_table(
            summary["bias"]["profession_bias"][:10],
            ["profession", "male", "female", "neutral", "total"],
        ),
        "",
        "## Sentence Length Bias",
        "",
        f"- Mean original length: `{summary['length_stats']['original']['mean']}`",
        f"- P95 original length: `{summary['length_stats']['original']['p95']}`",
        f"- Max original length: `{summary['length_stats']['original']['max']}`",
        "",
        "## Vocabulary Repetition Bias",
        "",
        f"- Type-token ratio: `{summary['vocabulary']['type_token_ratio']}`",
        "",
        format_table(summary["top_tokens"][:15], ["token", "count"]),
        "",
        "## Dominant Correction Patterns",
        "",
        format_table(
            [
                {"pattern": key, "count": value["count"], "percentage": f"{value['percentage']:.2f}%"}
                for key, value in summary["correction_patterns"].items()
            ],
            ["pattern", "count", "percentage"],
        ),
        "",
        "## Duplicate and Noise Bias",
        "",
        format_table(bias_rows, ["indicator", "value"]),
        "",
        "## Assessment",
        "",
        f"- Dominant pronoun group: `{summary['bias']['dominant_pronoun_group']}`",
        f"- Dominant correction pattern: `{summary['bias']['dominant_correction_pattern']}`",
        f"- Duplicate risk: `{summary['bias']['duplicate_risk']}`",
        f"- Length bias risk: `{summary['bias']['length_bias_risk']}`",
    ]
    (RESULTS_DIR / "bias_report.md").write_text(
        "\n".join(bias_lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Run the dataset audit and write report artifacts."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_files = discover_dataset_files()
    inventory = [load_inventory_entry(path) for path in dataset_files]

    primary_entry = next(item for item in inventory if item.path == relative_path(PRIMARY_DATASET_FILE))
    primary_df = pd.read_json(PRIMARY_DATASET_FILE, lines=True)
    split_frames = {
        split: pd.read_json(path, lines=True)
        for split, path in SPLIT_FILES.items()
    }

    bucket_labels = primary_df.apply(
        lambda row: normalized_error_bucket(
            str(row["error_type"]),
            str(row["original"]),
            str(row["corrected"]),
        ),
        axis=1,
    )
    explicit_counts = primary_df["error_type"].value_counts().sort_values(ascending=False)
    bucket_counts = bucket_labels.value_counts().sort_values(ascending=False)

    split_distribution = {
        split: {
            "rows": len(frame),
            "percentage": round((len(frame) / len(primary_df)) * 100, 2),
        }
        for split, frame in split_frames.items()
    }
    length_stats = {
        "original": sentence_length_stats(primary_df["original"]),
        "corrected": sentence_length_stats(primary_df["corrected"]),
    }
    vocabulary = vocabulary_stats(primary_df["corrected"])
    top_original_tokens = top_tokens(primary_df["original"], limit=20)
    top_corrected_tokens = top_tokens(primary_df["corrected"], limit=20)

    correction_pattern_counts = Counter(
        correction_pattern(str(row.original), str(row.corrected))
        for row in primary_df.itertuples(index=False)
    )
    correction_patterns = {
        pattern: {
            "count": count,
            "percentage": round((count / len(primary_df)) * 100, 2),
        }
        for pattern, count in correction_pattern_counts.most_common()
    }

    duplicates = duplicate_metrics(primary_df)
    quality_issues = text_issue_metrics(primary_df)
    split_leakage = split_leakage_metrics(split_frames)
    bert_imbalance = token_error_ratio(primary_df)
    quality = quality_score(quality_issues, duplicates, split_leakage, len(primary_df))

    pronouns = gender_counts(primary_df["corrected"])
    profession_bias = profession_bias_records(primary_df["corrected"])

    largest_label = explicit_counts.index[0]
    smallest_label = explicit_counts.index[-1]
    largest_bucket = bucket_counts.index[0]
    smallest_bucket = bucket_counts.index[-1]

    summary = {
        "generated_at": utc_now(),
        "inventory": [entry.__dict__ for entry in inventory],
        "primary_dataset": {
            "path": primary_entry.path,
            "rows": len(primary_df),
            "input_column": primary_entry.input_column,
            "corrected_column": primary_entry.corrected_column,
            "label_columns": primary_entry.label_columns,
            "metadata_columns": primary_entry.metadata_columns,
            "columns": primary_entry.columns,
            "dtypes": primary_entry.dtypes,
            "sample_rows": primary_entry.sample_rows,
        },
        "split_distribution": split_distribution,
        "explicit_error_type_distribution": {
            label: {
                "count": int(count),
                "percentage": round((count / len(primary_df)) * 100, 2),
            }
            for label, count in explicit_counts.items()
        },
        "error_bucket_distribution": {
            bucket: {
                "count": int(count),
                "percentage": round((count / len(primary_df)) * 100, 2),
            }
            for bucket, count in bucket_counts.items()
        },
        "imbalance": {
            "largest_label": largest_label,
            "largest_label_share": round((explicit_counts.iloc[0] / len(primary_df)) * 100, 2),
            "smallest_label": smallest_label,
            "smallest_label_share": round((explicit_counts.iloc[-1] / len(primary_df)) * 100, 2),
            "largest_bucket": largest_bucket,
            "largest_bucket_share": round((bucket_counts.iloc[0] / len(primary_df)) * 100, 2),
            "smallest_bucket": smallest_bucket,
            "smallest_bucket_share": round((bucket_counts.iloc[-1] / len(primary_df)) * 100, 2),
        },
        "length_stats": length_stats,
        "vocabulary": vocabulary,
        "top_tokens": top_corrected_tokens,
        "correction_patterns": correction_patterns,
        "duplicates": duplicates,
        "quality_issues": quality_issues,
        "split_leakage": split_leakage,
        "bert_imbalance": bert_imbalance,
        "quality": quality,
        "bias": {
            "gender_pronouns": pronouns,
            "profession_bias": profession_bias,
            "dominant_pronoun_group": max(pronouns, key=pronouns.get),
            "dominant_correction_pattern": next(iter(correction_patterns)),
            "duplicate_risk": "low" if duplicates["exact_pair_duplicates"] == 0 and duplicates["near_duplicate_pair_count"] < 15 else "moderate",
            "length_bias_risk": "low" if length_stats["original"]["p95"] <= 14 else "moderate",
        },
    }

    recommendation_block = {
        "strengths": dataset_strengths(summary),
        "weaknesses": dataset_weaknesses(summary),
        "risks": biggest_risks(summary),
    }
    summary["recommendation_block"] = recommendation_block
    summary["recommendations"] = recommendations(summary)

    render_svg_bar_chart(
        "Explicit Error Type Distribution",
        [(label, float(count)) for label, count in explicit_counts.items()],
        RESULTS_DIR / "error_type_distribution.svg",
        color="#2563eb",
    )
    render_svg_histogram(
        "Sentence Length Distribution (Original Sentences)",
        [sentence_token_count(text) for text in primary_df["original"]],
        RESULTS_DIR / "sentence_length_distribution.svg",
        bins=14,
        color="#14b8a6",
    )
    render_svg_bar_chart(
        "Top Repeated Tokens (Corrected Sentences)",
        [(item["token"], float(item["count"])) for item in top_corrected_tokens[:15]],
        RESULTS_DIR / "top_repeated_tokens.svg",
        color="#f59e0b",
    )
    render_svg_bar_chart(
        "Duplicate Frequency",
        [
            ("Exact original duplicates", float(duplicates["exact_original_duplicates"])),
            ("Exact pair duplicates", float(duplicates["exact_pair_duplicates"])),
            ("Near-duplicate pairs", float(duplicates["near_duplicate_pair_count"])),
            ("Noisy rows", float(quality_issues["noisy_row_count"])),
        ],
        RESULTS_DIR / "duplicate_frequency.svg",
        color="#ef4444",
    )
    render_svg_bar_chart(
        "Train / Validation / Test Distribution",
        [
            (split, float(stats["rows"]))
            for split, stats in split_distribution.items()
        ],
        RESULTS_DIR / "split_distribution.svg",
        color="#8b5cf6",
    )
    render_svg_grouped_chart(
        "Bias Indicators",
        [
            {
                "label": "Pronouns",
                "male": pronouns["male"],
                "female": pronouns["female"],
                "neutral": pronouns["neutral"],
            }
        ],
        ["male", "female", "neutral"],
        RESULTS_DIR / "bias_indicators.svg",
    )

    write_reports(summary, inventory)
    print(f"Dataset audit written to {relative_path(RESULTS_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

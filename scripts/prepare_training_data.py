"""Prepare cleaned and balanced training-ready grammar correction datasets."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import unicodedata
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

LOGGER = logging.getLogger("prepare_training_data")

DEFAULT_RAW_INPUT = PROJECT_ROOT / "data" / "raw" / "grammar_correction.csv"
DEFAULT_FULL_INPUT = (
    PROJECT_ROOT / "data" / "processed" / "grammar_correction_full.jsonl"
)
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "clean"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "data_preparation"
DEFAULT_SEED = 42

REQUIRED_COLUMNS = {"original", "corrected", "error_type"}
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2, "unassigned": 3}

COLUMN_ALIASES = {
    "serial number": "serial_number",
    "serial_number": "serial_number",
    "id": "serial_number",
    "error type": "error_type",
    "error_type": "error_type",
    "label": "error_type",
    "labels": "error_type",
    "ungrammatical statement": "original",
    "source sentence": "original",
    "source_text": "original",
    "original": "original",
    "input": "original",
    "standard english": "corrected",
    "target": "corrected",
    "corrected": "corrected",
    "reference": "corrected",
    "references": "references",
    "split": "split",
    "source": "source",
}

NO_ERROR_KEYWORDS = {
    "no error",
    "no_error",
    "clean",
    "correct sentence",
    "already correct",
    "grammatically correct",
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


@dataclass
class SplitLookup:
    """Lookup tables for existing split assignments."""

    serial_to_split: dict[int, str]
    pair_to_split: dict[str, str]


def utc_now() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""

    return datetime.now(timezone.utc).isoformat()


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render a markdown table without extra dependencies."""

    if not rows:
        return "| No rows |"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training-data preparation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=None, help="Raw CSV or full JSONL input file"
    )
    parser.add_argument(
        "--processed-dir",
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory containing existing processed split files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for cleaned dataset outputs",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory for preparation reports and charts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible oversampling",
    )
    parser.add_argument(
        "--max-class-size",
        type=int,
        default=None,
        help="Optional cap applied to oversampled class sizes",
    )
    parser.add_argument(
        "--no-oversample",
        action="store_true",
        help="Disable oversampling and write balanced_train as the clean train split",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure console logging for the preparation script."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def resolve_input_path(input_arg: str | None) -> Path:
    """Resolve the input dataset path."""

    if input_arg:
        return Path(input_arg).resolve()
    if DEFAULT_RAW_INPUT.exists():
        return DEFAULT_RAW_INPUT
    if DEFAULT_FULL_INPUT.exists():
        return DEFAULT_FULL_INPUT
    raise FileNotFoundError("Could not locate a default dataset input.")


def normalize_column_name(column: str) -> str:
    """Normalize a raw column name into the canonical schema."""

    normalized = " ".join(str(column).strip().lower().replace("_", " ").split())
    return COLUMN_ALIASES.get(normalized, normalized.replace(" ", "_"))


def load_table(path: Path) -> pd.DataFrame:
    """Load a CSV or JSONL table and normalize column names."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        dataframe = pd.read_csv(path)
    elif suffix == ".jsonl":
        dataframe = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported input type: {path}")

    dataframe = dataframe.rename(
        columns={column: normalize_column_name(column) for column in dataframe.columns}
    )
    return dataframe


def normalize_text(value: Any) -> str:
    """Normalize text by removing control characters and collapsing whitespace."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = "".join(char for char in text if char.isprintable() or char in "\t\n\r")
    text = " ".join(text.split())
    return text.strip()


def normalize_pair_key(original: str, corrected: str) -> str:
    """Build a normalized pair key for duplicates and split lookups."""

    def _normalize(text: str) -> str:
        lowered = text.lower()
        normalized = "".join(
            char for char in lowered if char.isalnum() or char.isspace()
        )
        return " ".join(normalized.split())

    return f"{_normalize(original)} || {_normalize(corrected)}"


def tokenize_whitespace(text: str) -> list[str]:
    """Tokenize text using simple whitespace splitting."""

    return [token for token in text.split() if token]


def tokenize_alpha(text: str) -> list[str]:
    """Tokenize text using simple alphanumeric extraction."""

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


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Ensure the required training columns exist."""

    missing = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def is_no_error_label(label: str) -> bool:
    """Return whether a label explicitly marks a clean sentence."""

    normalized = normalize_text(label).lower()
    return any(keyword in normalized for keyword in NO_ERROR_KEYWORDS)


def is_punctuation_only_change(original: str, corrected: str) -> bool:
    """Return whether two strings differ only in punctuation."""

    original_alpha = "".join(
        char for char in original.lower() if char.isalnum() or char.isspace()
    )
    corrected_alpha = "".join(
        char for char in corrected.lower() if char.isalnum() or char.isspace()
    )
    return original_alpha == corrected_alpha and original != corrected


def is_capitalization_only_change(original: str, corrected: str) -> bool:
    """Return whether two strings differ only by casing."""

    return original.lower() == corrected.lower() and original != corrected


def infer_normalized_error_type(original: str, corrected: str) -> str:
    """Infer a normalized error bucket from sentence differences."""

    if original == corrected:
        return "no_error"
    if is_capitalization_only_change(original, corrected):
        return "capitalization"
    if is_punctuation_only_change(original, corrected):
        return "punctuation"

    source_tokens = tokenize_alpha(original)
    target_tokens = tokenize_alpha(corrected)
    matcher = SequenceMatcher(a=source_tokens, b=target_tokens)
    source_changes: list[str] = []
    target_changes: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        source_changes.extend(source_tokens[i1:i2])
        target_changes.extend(target_tokens[j1:j2])

    changed_tokens = set(source_changes + target_changes)
    if changed_tokens & ARTICLES:
        return "article"
    if changed_tokens & PREPOSITIONS:
        return "preposition"

    subject_verb_pairs = {
        ("go", "goes"),
        ("goes", "go"),
        ("have", "has"),
        ("has", "have"),
        ("is", "are"),
        ("are", "is"),
        ("was", "were"),
        ("were", "was"),
        ("do", "does"),
        ("does", "do"),
        ("need", "needs"),
        ("needs", "need"),
    }
    for source_token, target_token in zip(source_tokens, target_tokens):
        if (source_token, target_token) in subject_verb_pairs:
            return "subject_verb_agreement"

    if len(source_changes) == len(target_changes) == 1:
        source_token = source_changes[0]
        target_token = target_changes[0]
        similarity = SequenceMatcher(a=source_token, b=target_token).ratio()
        if similarity >= 0.75:
            return "spelling"
        if source_token.endswith(("ed", "ing", "s")) or target_token.endswith(
            ("ed", "ing", "s")
        ):
            return "tense"

    if any(token.endswith(("ed", "ing")) for token in source_changes + target_changes):
        return "tense"
    return "mixed_multiple"


def normalize_error_type(label: Any, original: str, corrected: str) -> str:
    """Normalize a source label into the Sprint 8 bucket set."""

    label_text = normalize_text(label).lower()
    if not label_text:
        return infer_normalized_error_type(original, corrected)

    if is_no_error_label(label_text):
        return "no_error"
    if "subject-verb" in label_text or (
        "agreement" in label_text and "subject" in label_text
    ):
        return "subject_verb_agreement"
    if "article" in label_text:
        return "article"
    if (
        "tense" in label_text
        or "conditional" in label_text
        or "gerund" in label_text
        or "participle" in label_text
        or "infinitive" in label_text
        or "auxiliar" in label_text
    ):
        return "tense"
    if "spelling" in label_text or "typo" in label_text:
        return "spelling"
    if (
        "punctuation" in label_text
        or "ellipsis" in label_text
        or "run-on" in label_text
        or "contraction" in label_text
    ):
        return "punctuation"
    if "capitalization" in label_text or "capitalisation" in label_text:
        return "capitalization"
    if "preposition" in label_text:
        return "preposition"
    if label_text in {
        "ambiguity",
        "sentence structure errors",
        "sentence fragments",
        "relative clause errors",
        "parallelism errors",
        "lack of parallelism in lists or series",
        "modifiers misplacement",
        "conjunction misuse",
    }:
        return "mixed_multiple"
    if label_text in {
        "abbreviation errors",
        "agreement in comparative and superlative forms",
        "clichés",
        "cliches",
        "faulty comparisons",
        "inappropriate register",
        "mixed metaphors/idioms",
        "negation errors",
        "passive voice overuse",
        "pronoun errors",
        "quantifier errors",
        "redundancy/repetition",
        "slang, jargon, and colloquialisms",
        "tautology",
        "word choice/usage",
    }:
        return "other"
    return infer_normalized_error_type(original, corrected)


def build_split_lookup(processed_dir: Path) -> SplitLookup:
    """Load existing processed split files to preserve split assignments."""

    serial_to_split: dict[int, str] = {}
    pair_to_split: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        split_path = processed_dir / f"grammar_correction_{split}.jsonl"
        if not split_path.exists():
            continue
        split_frame = load_table(split_path)
        if "serial_number" in split_frame.columns:
            for value in split_frame["serial_number"].dropna().tolist():
                serial_to_split[int(value)] = split
        if {"original", "corrected"} <= set(split_frame.columns):
            for row in split_frame.itertuples(index=False):
                pair_key = normalize_pair_key(
                    normalize_text(row.original), normalize_text(row.corrected)
                )
                pair_to_split.setdefault(pair_key, split)
    return SplitLookup(serial_to_split=serial_to_split, pair_to_split=pair_to_split)


def assign_splits(
    dataframe: pd.DataFrame,
    split_lookup: SplitLookup,
    seed: int,
) -> pd.DataFrame:
    """Assign train/validation/test splits to all rows."""

    assigned = dataframe.copy()
    if "split" in assigned.columns:
        assigned["split"] = assigned["split"].fillna("").map(normalize_text).str.lower()
    else:
        assigned["split"] = ""

    if "serial_number" in assigned.columns:
        assigned["serial_number"] = pd.to_numeric(
            assigned["serial_number"], errors="coerce"
        ).astype("Int64")
    else:
        assigned["serial_number"] = pd.Series(
            range(1, len(assigned) + 1), dtype="Int64"
        )

    pair_keys = assigned.apply(
        lambda row: normalize_pair_key(row["original"], row["corrected"]),
        axis=1,
    )
    assigned["pair_key"] = pair_keys

    def resolve_split(row: pd.Series) -> str:
        raw_split = row.get("split", "")
        if raw_split in SPLIT_ORDER:
            return raw_split
        serial_value = row.get("serial_number")
        if pd.notna(serial_value) and int(serial_value) in split_lookup.serial_to_split:
            return split_lookup.serial_to_split[int(serial_value)]
        pair_key = row["pair_key"]
        if pair_key in split_lookup.pair_to_split:
            return split_lookup.pair_to_split[pair_key]
        return "unassigned"

    assigned["split"] = assigned.apply(resolve_split, axis=1)

    unassigned_index = assigned.index[assigned["split"] == "unassigned"].tolist()
    if unassigned_index:
        LOGGER.info(
            "Assigning %s previously unassigned rows with a deterministic split.",
            len(unassigned_index),
        )
        rng = random.Random(seed)
        shuffled = list(unassigned_index)
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_cutoff = int(total * 0.8)
        validation_cutoff = train_cutoff + int(total * 0.1)
        for idx, row_index in enumerate(shuffled):
            if idx < train_cutoff:
                assigned.at[row_index, "split"] = "train"
            elif idx < validation_cutoff:
                assigned.at[row_index, "split"] = "validation"
            else:
                assigned.at[row_index, "split"] = "test"
    return assigned


def normalize_dataset_columns(
    dataframe: pd.DataFrame, source_name: str
) -> pd.DataFrame:
    """Normalize and enrich the primary dataframe."""

    normalized = dataframe.rename(
        columns={column: normalize_column_name(column) for column in dataframe.columns}
    ).copy()
    validate_required_columns(normalized)

    normalized["original"] = normalized["original"].map(normalize_text)
    normalized["corrected"] = normalized["corrected"].map(normalize_text)
    normalized["error_type"] = normalized["error_type"].fillna("").map(normalize_text)
    if "references" not in normalized.columns:
        normalized["references"] = normalized["corrected"].map(
            lambda value: [value] if value else []
        )
    else:
        normalized["references"] = normalized["references"].map(
            lambda value: (
                value
                if isinstance(value, list)
                else ([normalize_text(value)] if normalize_text(value) else [])
            )
        )
    if "source" not in normalized.columns:
        normalized["source"] = source_name
    else:
        normalized["source"] = (
            normalized["source"].fillna(source_name).map(normalize_text)
        )

    if "serial_number" not in normalized.columns:
        normalized["serial_number"] = pd.Series(
            range(1, len(normalized) + 1), dtype="Int64"
        )

    normalized["normalized_error_type"] = normalized.apply(
        lambda row: normalize_error_type(
            row["error_type"], row["original"], row["corrected"]
        ),
        axis=1,
    )
    return normalized


def detect_near_duplicates(dataframe: pd.DataFrame) -> tuple[int, list[dict[str, Any]]]:
    """Detect near-duplicate source-target pairs heuristically."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataframe.itertuples(index=False):
        original_tokens = tokenize_alpha(row.original)
        first_token = original_tokens[0] if original_tokens else "<empty>"
        buckets[first_token].append(
            {
                "serial_number": (
                    int(row.serial_number) if pd.notna(row.serial_number) else -1
                ),
                "original": row.original,
                "corrected": row.corrected,
                "normalized_error_type": row.normalized_error_type,
                "token_set": set(original_tokens),
                "token_length": len(set(original_tokens)),
            }
        )

    examples: list[dict[str, Any]] = []
    count = 0
    for bucket_records in buckets.values():
        for index, record_a in enumerate(bucket_records):
            for record_b in bucket_records[index + 1 :]:
                if normalize_pair_key(
                    record_a["original"], record_a["corrected"]
                ) == normalize_pair_key(record_b["original"], record_b["corrected"]):
                    continue
                if abs(record_a["token_length"] - record_b["token_length"]) > 2:
                    continue
                union = record_a["token_set"] | record_b["token_set"]
                if not union:
                    continue
                jaccard = len(record_a["token_set"] & record_b["token_set"]) / len(
                    union
                )
                if jaccard < 0.8:
                    continue
                similarity = SequenceMatcher(
                    a=record_a["original"].lower(),
                    b=record_b["original"].lower(),
                ).ratio()
                if similarity < 0.92:
                    continue
                count += 1
                if len(examples) < 15:
                    examples.append(
                        {
                            "serial_a": record_a["serial_number"],
                            "serial_b": record_b["serial_number"],
                            "normalized_error_type_a": record_a[
                                "normalized_error_type"
                            ],
                            "normalized_error_type_b": record_b[
                                "normalized_error_type"
                            ],
                            "original_a": record_a["original"],
                            "original_b": record_b["original"],
                            "similarity": round(similarity, 3),
                        }
                    )
    return count, examples


def compute_length_stats(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Compute sentence-length statistics."""

    original_lengths = [
        len(tokenize_whitespace(text)) for text in dataframe["original"]
    ]
    corrected_lengths = [
        len(tokenize_whitespace(text)) for text in dataframe["corrected"]
    ]

    def _stats(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
        return {
            "mean": round(float(np.mean(values)), 3),
            "median": round(float(np.median(values)), 3),
            "min": int(np.min(values)),
            "max": int(np.max(values)),
        }

    return {
        "original": _stats(original_lengths),
        "corrected": _stats(corrected_lengths),
    }


def distribution_counts(dataframe: pd.DataFrame, column: str) -> dict[str, int]:
    """Return counts for a categorical column."""

    if dataframe.empty:
        return {}
    counts = dataframe[column].value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def summarize_dataset(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Summarize core statistics for a dataset frame."""

    split_counts = (
        distribution_counts(dataframe, "split") if "split" in dataframe.columns else {}
    )
    return {
        "rows": int(len(dataframe)),
        "normalized_error_type_distribution": distribution_counts(
            dataframe, "normalized_error_type"
        ),
        "length_stats": compute_length_stats(dataframe),
        "split_counts": split_counts,
    }


def annotate_issue_rows(
    dataframe: pd.DataFrame,
    reason: str,
    action: str,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Attach reason metadata to removed or flagged rows."""

    if dataframe.empty:
        return dataframe.copy()
    annotated = dataframe.copy()
    annotated["issue_reason"] = reason
    annotated["issue_action"] = action
    if extra:
        for key, value in extra.items():
            annotated[key] = value
    return annotated


def clean_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Clean the full dataset and return clean, removed, and flagged frames."""

    working = dataframe.copy()
    working["pair_key"] = working.apply(
        lambda row: normalize_pair_key(row["original"], row["corrected"]), axis=1
    )
    working["record_id"] = working.apply(
        lambda row: (
            f"serial_{int(row['serial_number'])}"
            if pd.notna(row["serial_number"])
            else f"row_{row.name}"
        ),
        axis=1,
    )
    working["split_priority"] = working["split"].map(SPLIT_ORDER).fillna(3).astype(int)

    removed_frames: list[pd.DataFrame] = []
    flagged_frames: list[pd.DataFrame] = []
    counters = defaultdict(int)

    missing_mask = working["original"].eq("") | working["corrected"].eq("")
    removed_missing = annotate_issue_rows(
        working.loc[missing_mask],
        reason="missing_original_or_corrected_text",
        action="removed",
    )
    counters["rows_removed"] += int(len(removed_missing))
    counters["removed_empty_rows"] += int(len(removed_missing))
    removed_frames.append(removed_missing)
    working = working.loc[~missing_mask].copy()

    unchanged_mask = working["original"].eq(working["corrected"])
    counters["original_equals_corrected_count"] = int(unchanged_mask.sum())
    keep_no_error_mask = unchanged_mask & working["normalized_error_type"].eq(
        "no_error"
    )
    flagged_unchanged = annotate_issue_rows(
        working.loc[unchanged_mask & ~keep_no_error_mask],
        reason="original_equals_corrected_without_no_error_label",
        action="flagged",
    )
    counters["rows_flagged"] += int(len(flagged_unchanged))
    counters["flagged_original_equals_corrected"] += int(len(flagged_unchanged))
    flagged_frames.append(flagged_unchanged)
    working = working.loc[~(unchanged_mask & ~keep_no_error_mask)].copy()

    working = working.sort_values(
        by=["split_priority", "serial_number", "record_id"],
        kind="stable",
    ).reset_index(drop=True)
    seen_pairs: dict[str, tuple[str, str]] = {}
    keep_indices: list[int] = []
    duplicate_rows: list[dict[str, Any]] = []

    for row_index, row in working.iterrows():
        pair_key = row["pair_key"]
        if pair_key not in seen_pairs:
            seen_pairs[pair_key] = (row["split"], row["record_id"])
            keep_indices.append(row_index)
            continue
        kept_split, kept_record_id = seen_pairs[pair_key]
        if row["split"] != kept_split:
            reason = f"split_leakage_duplicate_kept_in_{kept_split}"
            counters["leakage_rows_removed"] += 1
        else:
            reason = "exact_duplicate_pair"
            counters["exact_duplicates_removed"] += 1
        row_data = row.to_dict()
        row_data["issue_reason"] = reason
        row_data["issue_action"] = "removed"
        row_data["duplicate_of_record_id"] = kept_record_id
        row_data["duplicate_of_split"] = kept_split
        duplicate_rows.append(row_data)

    removed_duplicates = pd.DataFrame(duplicate_rows)
    counters["rows_removed"] += int(len(removed_duplicates))
    if not removed_duplicates.empty:
        removed_frames.append(removed_duplicates)

    clean = working.iloc[keep_indices].copy().reset_index(drop=True)
    removed = (
        pd.concat(removed_frames, ignore_index=True, sort=False)
        if removed_frames
        else pd.DataFrame(
            columns=list(working.columns) + ["issue_reason", "issue_action"]
        )
    )
    flagged = (
        pd.concat(flagged_frames, ignore_index=True, sort=False)
        if flagged_frames
        else pd.DataFrame(
            columns=list(working.columns) + ["issue_reason", "issue_action"]
        )
    )
    return clean, removed, flagged, dict(counters)


def oversample_training_data(
    train_frame: pd.DataFrame,
    seed: int,
    max_class_size: int | None,
    enable_oversampling: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a balanced training dataframe via random oversampling."""

    base = train_frame.copy().reset_index(drop=True)
    base["record_id"] = base.apply(
        lambda row: row.get("record_id") or f"train_{int(row['serial_number'])}",
        axis=1,
    )
    base["is_oversampled"] = False
    base["oversample_source_id"] = None

    counts = base["normalized_error_type"].value_counts().to_dict()
    non_no_error_counts = {
        label: count for label, count in counts.items() if label != "no_error"
    }
    majority_count = max(
        non_no_error_counts.values(), default=max(counts.values(), default=0)
    )
    target_count = (
        min(majority_count, max_class_size) if max_class_size else majority_count
    )

    rng = random.Random(seed)
    balanced_parts: list[pd.DataFrame] = []
    oversample_counts: dict[str, int] = {}

    for label, group in base.groupby("normalized_error_type", sort=True):
        group = group.copy().reset_index(drop=True)
        current_count = len(group)
        if label == "no_error":
            desired_count = (
                min(current_count, target_count) if target_count else current_count
            )
        else:
            desired_count = target_count or current_count

        if max_class_size and current_count > desired_count:
            sampled_indices = rng.sample(list(range(current_count)), desired_count)
            selected = group.iloc[sampled_indices].copy().reset_index(drop=True)
            oversample_counts[label] = 0
            balanced_parts.append(selected)
            continue

        balanced_parts.append(group)
        oversample_counts[label] = 0
        if (
            not enable_oversampling
            or current_count >= desired_count
            or current_count == 0
        ):
            continue

        needed = desired_count - current_count
        sampled_indices = [rng.randrange(current_count) for _ in range(needed)]
        sampled = group.iloc[sampled_indices].copy().reset_index(drop=True)
        sampled["is_oversampled"] = True
        sampled["oversample_source_id"] = sampled["record_id"]
        sampled["record_id"] = [
            f"{source_id}__oversampled_{index + 1}"
            for index, source_id in enumerate(sampled["oversample_source_id"].tolist())
        ]
        balanced_parts.append(sampled)
        oversample_counts[label] = needed

    balanced = pd.concat(balanced_parts, ignore_index=True, sort=False)
    balanced = balanced.sort_values(
        by=["normalized_error_type", "is_oversampled", "serial_number", "record_id"],
        kind="stable",
    ).reset_index(drop=True)
    metadata = {
        "majority_count": int(majority_count),
        "target_count": int(target_count),
        "oversample_counts": oversample_counts,
    }
    return balanced, metadata


def generate_token_labels(
    original: str,
    corrected: str,
) -> tuple[list[str], list[int]]:
    """Generate whitespace-token labels from original/corrected diffs."""

    tokens = tokenize_whitespace(original)
    corrected_tokens = tokenize_whitespace(corrected)
    labels = [0 for _ in tokens]
    matcher = SequenceMatcher(a=tokens, b=corrected_tokens)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"replace", "delete"}:
            for index in range(i1, min(i2, len(labels))):
                labels[index] = 1
        elif tag == "insert":
            anchor = i1 if i1 < len(labels) else len(labels) - 1
            if anchor >= 0:
                labels[anchor] = 1
    return tokens, labels


def build_bert_token_label_frame(
    train_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create token-level labels for BERT token classification."""

    rows: list[dict[str, Any]] = []
    total_tokens = 0
    error_tokens = 0
    correct_tokens = 0

    for row in train_frame.itertuples(index=False):
        tokens, labels = generate_token_labels(row.original, row.corrected)
        error_count = sum(labels)
        correct_count = len(labels) - error_count
        total_tokens += len(labels)
        error_tokens += error_count
        correct_tokens += correct_count
        rows.append(
            {
                "record_id": row.record_id,
                "serial_number": (
                    int(row.serial_number) if pd.notna(row.serial_number) else None
                ),
                "original": row.original,
                "corrected": row.corrected,
                "tokens": tokens,
                "labels": labels,
                "normalized_error_type": row.normalized_error_type,
                "split": row.split,
            }
        )

    summary = {
        "total_examples": len(rows),
        "total_tokens": int(total_tokens),
        "correct_token_count": int(correct_tokens),
        "error_token_count": int(error_tokens),
        "correct_to_error_ratio": round(correct_tokens / max(error_tokens, 1), 4),
    }
    return pd.DataFrame(rows), summary


def write_jsonl(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to JSON Lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    export_frame = dataframe.drop(
        columns=["pair_key", "split_priority"],
        errors="ignore",
    )
    records = export_frame.replace({np.nan: None}).to_dict(orient="records")
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_svg_bar_chart(
    title: str,
    items: list[tuple[str, float]],
    output_path: Path,
    color: str = "#2563eb",
) -> None:
    """Render a simple horizontal SVG bar chart."""

    if not items:
        output_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8"
        )
        return

    width = 1100
    label_width = 300
    value_width = 70
    bar_height = 26
    top_padding = 70
    left_padding = 28
    chart_width = width - label_width - value_width - 80
    max_value = max(value for _, value in items) or 1.0
    height = top_padding + len(items) * (bar_height + 10) + 30

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#111827;} .title{font-size:20px;font-weight:bold;} .value{font-weight:bold;}</style>",
        f"<text class='title' x='{left_padding}' y='32'>{title}</text>",
    ]
    y = top_padding
    for label, value in items:
        bar_width = (value / max_value) * chart_width if max_value else 0
        lines.append(f"<text x='{left_padding}' y='{y + 17}'>{label}</text>")
        lines.append(
            f"<rect x='{left_padding + label_width}' y='{y}' width='{chart_width}' height='{bar_height}' rx='4' fill='#e5e7eb' />"
        )
        lines.append(
            f"<rect x='{left_padding + label_width}' y='{y}' width='{bar_width:.2f}' height='{bar_height}' rx='4' fill='{color}' />"
        )
        lines.append(
            f"<text class='value' x='{left_padding + label_width + chart_width + 12}' y='{y + 17}'>{value:.0f}</text>"
        )
        y += bar_height + 10
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_svg_histogram(
    title: str,
    values: list[int],
    output_path: Path,
    bins: int = 12,
    color: str = "#14b8a6",
) -> None:
    """Render a simple SVG histogram."""

    width = 1100
    height = 500
    margin = {"left": 60, "right": 30, "top": 70, "bottom": 80}
    chart_width = width - margin["left"] - margin["right"]
    chart_height = height - margin["top"] - margin["bottom"]

    if not values:
        output_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8"
        )
        return

    if min(values) == max(values):
        counts = np.array([len(values)])
        edges = np.array([min(values), max(values) + 1])
    else:
        counts, edges = np.histogram(values, bins=bins)
    max_count = int(max(counts)) or 1
    bar_count = len(counts)
    bar_width = chart_width / max(bar_count, 1)

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#111827;} .title{font-size:20px;font-weight:bold;} .axis{stroke:#9ca3af;stroke-width:1;} .tick{fill:#6b7280;}</style>",
        f"<text class='title' x='{margin['left']}' y='32'>{title}</text>",
        f"<line class='axis' x1='{margin['left']}' y1='{margin['top'] + chart_height}' x2='{margin['left'] + chart_width}' y2='{margin['top'] + chart_height}' />",
        f"<line class='axis' x1='{margin['left']}' y1='{margin['top']}' x2='{margin['left']}' y2='{margin['top'] + chart_height}' />",
    ]
    for index, count in enumerate(counts):
        x = margin["left"] + index * bar_width + 4
        bar_height = (count / max_count) * chart_height if max_count else 0
        y = margin["top"] + chart_height - bar_height
        lines.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{max(bar_width - 8, 2):.2f}' height='{bar_height:.2f}' fill='{color}' rx='3' />"
        )
        label = f"{int(edges[index])}-{int(edges[index + 1])}"
        lines.append(
            f"<text class='tick' x='{x + (bar_width - 8) / 2:.2f}' y='{margin['top'] + chart_height + 18}' text-anchor='middle'>{label}</text>"
        )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_class_distribution_frame(
    before_full: pd.DataFrame,
    clean_full: pd.DataFrame,
    clean_train: pd.DataFrame,
    balanced_train: pd.DataFrame,
) -> pd.DataFrame:
    """Build a before/after class distribution comparison frame."""

    labels = sorted(
        set(before_full["normalized_error_type"])
        | set(clean_full["normalized_error_type"])
        | set(clean_train["normalized_error_type"])
        | set(balanced_train["normalized_error_type"])
    )
    before_counts = Counter(before_full["normalized_error_type"].tolist())
    clean_counts = Counter(clean_full["normalized_error_type"].tolist())
    clean_train_counts = Counter(clean_train["normalized_error_type"].tolist())
    balanced_counts = Counter(balanced_train["normalized_error_type"].tolist())

    rows = []
    for label in labels:
        rows.append(
            {
                "normalized_error_type": label,
                "before_full": before_counts.get(label, 0),
                "after_clean_full": clean_counts.get(label, 0),
                "clean_train": clean_train_counts.get(label, 0),
                "balanced_train": balanced_counts.get(label, 0),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    report_path: Path,
    source_files: list[Path],
    before_frame: pd.DataFrame,
    clean_full: pd.DataFrame,
    clean_splits: dict[str, pd.DataFrame],
    balanced_train: pd.DataFrame,
    removed_rows: pd.DataFrame,
    flagged_rows: pd.DataFrame,
    counters: dict[str, int],
    near_duplicates_before: tuple[int, list[dict[str, Any]]],
    near_duplicates_after: tuple[int, list[dict[str, Any]]],
    oversampling_metadata: dict[str, Any],
    bert_summary: dict[str, Any],
    class_distribution_frame: pd.DataFrame,
) -> None:
    """Write the Sprint 8 markdown report."""

    before_stats = summarize_dataset(before_frame)
    after_stats = summarize_dataset(clean_full)
    clean_train = clean_splits["train"]
    clean_validation = clean_splits["validation"]
    clean_test = clean_splits["test"]

    removed_reasons = (
        removed_rows["issue_reason"].value_counts().to_dict()
        if not removed_rows.empty
        else {}
    )
    flagged_reasons = (
        flagged_rows["issue_reason"].value_counts().to_dict()
        if not flagged_rows.empty
        else {}
    )

    report_lines = [
        "# Sprint 8 Data Preparation Report",
        "",
        "## Executive Summary",
        "",
        f"- Generated: `{utc_now()}`",
        f"- Input rows before cleaning: `{before_stats['rows']}`",
        f"- Clean rows after cleaning: `{after_stats['rows']}`",
        f"- Clean train / validation / test: `{len(clean_train)}` / `{len(clean_validation)}` / `{len(clean_test)}`",
        f"- Balanced train rows: `{len(balanced_train)}`",
        f"- BERT correct:error token ratio: `{bert_summary['correct_to_error_ratio']}`",
        "",
        "## Files Used",
        "",
        *[f"- `{str(path)}`" for path in source_files],
        "",
        "## Cleaning Actions Performed",
        "",
        "- Removed rows with empty or null original/corrected text.",
        "- Normalized whitespace, unicode, and control characters.",
        "- Preserved clean/no_error unchanged rows and flagged unchanged non-no_error rows.",
        "- Removed exact duplicate source-target pairs.",
        "- Removed split leakage by keeping the earliest pair in train, then validation, then test.",
        "- Added normalized_error_type to all new clean exports.",
        "",
        "## Rows Removed and Why",
        "",
        f"- Rows removed total: `{len(removed_rows)}`",
        *[f"- `{reason}`: `{count}`" for reason, count in removed_reasons.items()],
        "",
        "## Rows Flagged and Why",
        "",
        f"- Rows flagged total: `{len(flagged_rows)}`",
        *[f"- `{reason}`: `{count}`" for reason, count in flagged_reasons.items()],
        "",
        "## Duplicate and Leakage Summary",
        "",
        f"- Exact duplicates removed: `{counters.get('exact_duplicates_removed', 0)}`",
        f"- Leakage duplicates removed: `{counters.get('leakage_rows_removed', 0)}`",
        f"- Near duplicates detected before cleaning: `{near_duplicates_before[0]}`",
        f"- Near duplicates detected after cleaning: `{near_duplicates_after[0]}`",
        "",
        "## Before / After Class Distribution",
        "",
        markdown_table(
            class_distribution_frame.to_dict(orient="records"),
            list(class_distribution_frame.columns),
        ),
        "",
        "## Before / After Dataset Stats",
        "",
        f"- Average original length before: `{before_stats['length_stats']['original']['mean']}` tokens",
        f"- Average corrected length before: `{before_stats['length_stats']['corrected']['mean']}` tokens",
        f"- Average original length after: `{after_stats['length_stats']['original']['mean']}` tokens",
        f"- Average corrected length after: `{after_stats['length_stats']['corrected']['mean']}` tokens",
        "",
        "## Oversampling Strategy",
        "",
        f"- Oversampling enabled: `{len(balanced_train) != len(clean_train)}`",
        f"- Majority class size used as target: `{oversampling_metadata['majority_count']}`",
        f"- Effective target class size: `{oversampling_metadata['target_count']}`",
        "- Validation and test splits were not oversampled.",
        "",
        "### Oversample Counts",
        "",
        markdown_table(
            [
                {
                    "normalized_error_type": label,
                    "oversampled_rows_added": count,
                }
                for label, count in sorted(
                    oversampling_metadata["oversample_counts"].items()
                )
            ],
            ["normalized_error_type", "oversampled_rows_added"],
        ),
        "",
        "## BERT Token-Label Summary",
        "",
        f"- Total examples: `{bert_summary['total_examples']}`",
        f"- Total tokens: `{bert_summary['total_tokens']}`",
        f"- Correct tokens: `{bert_summary['correct_token_count']}`",
        f"- Error tokens: `{bert_summary['error_token_count']}`",
        f"- Correct:error ratio: `{bert_summary['correct_to_error_ratio']}`",
        "",
        "## Training Readiness Conclusion",
        "",
        "- The cleaned corpus is structurally ready for T5 and BERT preprocessing.",
        "- Balanced training data now reduces class skew at the sentence-label level.",
        "- BERT token labels are still weak labels derived from sentence diffs, so weighted loss is still recommended in the next sprint.",
        "",
        "## Recommended Next Sprint",
        "",
        "- Use the cleaned and balanced exports for T5 and BERT training only after reviewing the flagged noisy rows.",
        "- Keep external datasets such as JFLEG or CoNLL on the roadmap to improve generalization beyond this relatively small corpus.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def prepare_training_data(
    input_path: Path,
    processed_dir: Path,
    output_dir: Path,
    results_dir: Path,
    seed: int,
    max_class_size: int | None = None,
    oversample: bool = True,
) -> dict[str, Any]:
    """Prepare cleaned and balanced training-ready datasets."""

    LOGGER.info("Loading input dataset from %s", input_path)
    raw_frame = load_table(input_path)
    normalized = normalize_dataset_columns(raw_frame, source_name=input_path.name)
    split_lookup = build_split_lookup(processed_dir)
    normalized = assign_splits(normalized, split_lookup, seed)
    before_frame = normalized.copy()

    near_duplicates_before = detect_near_duplicates(before_frame)
    clean_full, removed_rows, flagged_rows, counters = clean_dataset(normalized)
    near_duplicates_after = detect_near_duplicates(clean_full)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    clean_splits = {
        split: clean_full.loc[clean_full["split"] == split]
        .copy()
        .reset_index(drop=True)
        for split in ("train", "validation", "test")
    }

    balanced_train, oversampling_metadata = oversample_training_data(
        clean_splits["train"],
        seed=seed,
        max_class_size=max_class_size,
        enable_oversampling=oversample,
    )
    bert_labels_frame, bert_summary = build_bert_token_label_frame(
        clean_splits["train"]
    )

    clean_files = {
        "full": output_dir / "grammar_correction_clean_full.jsonl",
        "train": output_dir / "grammar_correction_clean_train.jsonl",
        "validation": output_dir / "grammar_correction_clean_validation.jsonl",
        "test": output_dir / "grammar_correction_clean_test.jsonl",
        "balanced_train": output_dir / "grammar_correction_balanced_train.jsonl",
        "bert_labels": output_dir / "bert_token_labels_train.jsonl",
        "removed_rows": output_dir / "removed_rows.jsonl",
        "flagged_rows": output_dir / "flagged_noisy_rows.jsonl",
    }

    write_jsonl(clean_full, clean_files["full"])
    write_jsonl(clean_splits["train"], clean_files["train"])
    write_jsonl(clean_splits["validation"], clean_files["validation"])
    write_jsonl(clean_splits["test"], clean_files["test"])
    write_jsonl(balanced_train, clean_files["balanced_train"])
    write_jsonl(bert_labels_frame, clean_files["bert_labels"])
    write_jsonl(removed_rows, clean_files["removed_rows"])
    write_jsonl(flagged_rows, clean_files["flagged_rows"])

    class_distribution_frame = build_class_distribution_frame(
        before_frame, clean_full, clean_splits["train"], balanced_train
    )
    class_distribution_frame.to_csv(
        results_dir / "class_distribution_before_after.csv", index=False
    )

    bert_summary_path = results_dir / "bert_token_label_summary.json"
    bert_summary_path.write_text(
        json.dumps(bert_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    render_svg_bar_chart(
        "Error Type Distribution Before Cleaning",
        [
            (label, float(count))
            for label, count in sorted(
                Counter(before_frame["normalized_error_type"]).items()
            )
        ],
        results_dir / "error_type_distribution_before.svg",
        color="#2563eb",
    )
    render_svg_bar_chart(
        "Error Type Distribution After Cleaning",
        [
            (label, float(count))
            for label, count in sorted(
                Counter(clean_full["normalized_error_type"]).items()
            )
        ],
        results_dir / "error_type_distribution_after_cleaning.svg",
        color="#0f766e",
    )
    render_svg_bar_chart(
        "Error Type Distribution After Balancing",
        [
            (label, float(count))
            for label, count in sorted(
                Counter(balanced_train["normalized_error_type"]).items()
            )
        ],
        results_dir / "error_type_distribution_after_balancing.svg",
        color="#7c3aed",
    )
    render_svg_histogram(
        "Sentence Length Distribution (Cleaned Full Dataset)",
        [len(tokenize_whitespace(text)) for text in clean_full["original"]],
        results_dir / "sentence_length_distribution_cleaned.svg",
        bins=14,
        color="#14b8a6",
    )
    render_svg_bar_chart(
        "BERT Token Label Distribution",
        [
            ("CORRECT", float(bert_summary["correct_token_count"])),
            ("ERROR", float(bert_summary["error_token_count"])),
        ],
        results_dir / "bert_token_label_distribution.svg",
        color="#ef4444",
    )

    report_path = results_dir / "sprint8_data_preparation_report.md"
    source_files = [
        input_path,
        processed_dir / "grammar_correction_train.jsonl",
        processed_dir / "grammar_correction_validation.jsonl",
        processed_dir / "grammar_correction_test.jsonl",
    ]
    source_files = [path for path in source_files if path.exists()]
    write_report(
        report_path=report_path,
        source_files=source_files,
        before_frame=before_frame,
        clean_full=clean_full,
        clean_splits=clean_splits,
        balanced_train=balanced_train,
        removed_rows=removed_rows,
        flagged_rows=flagged_rows,
        counters=counters,
        near_duplicates_before=near_duplicates_before,
        near_duplicates_after=near_duplicates_after,
        oversampling_metadata=oversampling_metadata,
        bert_summary=bert_summary,
        class_distribution_frame=class_distribution_frame,
    )

    summary = {
        "input_path": str(input_path),
        "cleaned_full_count": int(len(clean_full)),
        "clean_train_count": int(len(clean_splits["train"])),
        "balanced_train_count": int(len(balanced_train)),
        "validation_count": int(len(clean_splits["validation"])),
        "test_count": int(len(clean_splits["test"])),
        "bert_correct_to_error_ratio": bert_summary["correct_to_error_ratio"],
        "report_path": str(report_path),
        "output_dir": str(output_dir),
        "results_dir": str(results_dir),
    }
    return summary


def main() -> int:
    """CLI entrypoint for Sprint 8 data preparation."""

    configure_logging()
    args = parse_args()

    input_path = resolve_input_path(args.input)
    processed_dir = Path(args.processed_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    results_dir = Path(args.results_dir).resolve()

    summary = prepare_training_data(
        input_path=input_path,
        processed_dir=processed_dir,
        output_dir=output_dir,
        results_dir=results_dir,
        seed=args.seed,
        max_class_size=args.max_class_size,
        oversample=not args.no_oversample,
    )

    print(f"cleaned full count: {summary['cleaned_full_count']}")
    print(f"clean train count: {summary['clean_train_count']}")
    print(f"balanced train count: {summary['balanced_train_count']}")
    print(f"validation count: {summary['validation_count']}")
    print(f"test count: {summary['test_count']}")
    print(f"BERT correct:error token ratio: {summary['bert_correct_to_error_ratio']}")
    print(f"report: {summary['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

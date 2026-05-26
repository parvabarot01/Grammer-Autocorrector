"""Train the BERT grammar detector on processed JSONL data."""

from __future__ import annotations

import argparse
import json
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.models.bert_detector import BERTGrammarDetector
from src.utils.config import BASE_DIR, load_config


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for BERT fine-tuning."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--data_dir", default=str(BASE_DIR / "data" / "processed"))
    parser.add_argument("--output_dir", default=str(BASE_DIR / "models" / "bert"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_labels", type=int, default=2)
    return parser.parse_args()


def configure_logging(log_file: Path) -> None:
    """Configure console and file logging for detector training."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def resolve_dataset_paths(data_dir: Path) -> Tuple[Path, Path]:
    """Resolve train and validation dataset files from a processed directory."""

    preferred_prefixes = ["jfleg", "grammar_correction"]
    for prefix in preferred_prefixes:
        train_path = data_dir / f"{prefix}_train.jsonl"
        validation_path = data_dir / f"{prefix}_validation.jsonl"
        if train_path.exists() and validation_path.exists():
            return train_path, validation_path

    train_candidates = sorted(data_dir.glob("*_train.jsonl"))
    validation_candidates = sorted(data_dir.glob("*_validation.jsonl"))
    if train_candidates and validation_candidates:
        return train_candidates[0], validation_candidates[0]

    raise FileNotFoundError(
        f"Could not find processed train/validation splits in {data_dir}."
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON Lines file into a list of row dictionaries."""

    with path.open("r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def build_word_labels(original: str, corrected: str) -> List[int]:
    """Build binary word labels from source/target token differences."""

    original_tokens = original.split()
    corrected_tokens = corrected.split()
    labels = [0] * len(original_tokens)
    matcher = SequenceMatcher(
        a=[token.casefold() for token in original_tokens],
        b=[token.casefold() for token in corrected_tokens],
    )

    for operation, source_start, source_end, _, _ in matcher.get_opcodes():
        if operation == "equal":
            continue
        if operation == "insert":
            if source_start < len(labels):
                labels[source_start] = 1
            elif labels:
                labels[-1] = 1
            continue
        for index in range(source_start, min(source_end, len(labels))):
            labels[index] = 1

    return labels


def encode_examples(
    rows: List[Dict[str, Any]],
    detector: BERTGrammarDetector,
) -> List[Dict[str, Any]]:
    """Convert correction pairs into token classification training examples."""

    if detector.tokenizer is None:
        detector.load_model()

    encoded_rows: List[Dict[str, Any]] = []
    for row in rows:
        original = str(row["original"])
        corrected = str(row["corrected"])
        words = original.split()
        word_labels = build_word_labels(original, corrected)
        tokenized = detector.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=detector.max_length,
            return_attention_mask=True,
        )
        word_ids = tokenized.word_ids()
        labels: List[int] = []
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            else:
                labels.append(word_labels[word_id])

        encoded_row = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }
        encoded_row["token_type_ids"] = tokenized.get(
            "token_type_ids",
            [0] * len(tokenized["input_ids"]),
        )
        encoded_rows.append(encoded_row)

    return encoded_rows


def main() -> int:
    """Run BERT fine-tuning from processed JSONL files."""

    args = parse_args()
    configure_logging(BASE_DIR / "logs" / "train_bert.log")
    config = load_config()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path, validation_path = resolve_dataset_paths(data_dir)
    LOGGER.info("Using training data: %s", train_path)
    LOGGER.info("Using validation data: %s", validation_path)

    train_rows = load_jsonl(train_path)
    validation_rows = load_jsonl(validation_path)

    detector = BERTGrammarDetector(
        model_name=args.model_name,
        num_labels=args.num_labels,
        device=args.device,
    )
    detector.batch_size = args.batch_size
    detector.learning_rate = args.learning_rate
    detector.max_length = config.model.max_length
    detector.load_model()

    train_dataset = encode_examples(train_rows, detector)
    validation_dataset = encode_examples(validation_rows, detector)

    metrics = detector.fine_tune(
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        output_dir=str(output_dir),
        num_epochs=args.epochs,
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info(
        "Final validation metrics | precision=%.4f recall=%.4f f1=%.4f",
        float(metrics.get("eval_precision", metrics.get("precision", 0.0))),
        float(metrics.get("eval_recall", metrics.get("recall", 0.0))),
        float(metrics.get("eval_f1", metrics.get("f1", 0.0))),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

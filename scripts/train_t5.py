"""Train the T5 grammar corrector on processed JSONL data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.t5_corrector import T5GrammarCorrector
from src.utils.config import BASE_DIR, load_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for T5 fine-tuning."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="t5-base")
    parser.add_argument("--data_dir", default=str(BASE_DIR / "data" / "processed"))
    parser.add_argument("--output_dir", default=str(BASE_DIR / "models" / "t5"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def configure_logging(log_file: Path) -> None:
    """Configure console and file logging for training."""

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
    """Resolve train and validation JSONL paths from a processed data directory."""

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
    """Load a JSON Lines file into memory."""

    with path.open("r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def main() -> int:
    """Run T5 fine-tuning from processed JSONL files."""

    args = parse_args()
    configure_logging(BASE_DIR / "logs" / "train_t5.log")
    config = load_config()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path, validation_path = resolve_dataset_paths(data_dir)
    LOGGER.info("Using training data: %s", train_path)
    LOGGER.info("Using validation data: %s", validation_path)

    train_dataset = load_jsonl(train_path)
    validation_dataset = load_jsonl(validation_path)

    corrector = T5GrammarCorrector(model_name=args.model_name, device=args.device)
    corrector.batch_size = args.batch_size
    corrector.learning_rate = args.learning_rate
    corrector.num_beams = args.num_beams
    corrector.max_length = config.model.max_length

    metrics = corrector.fine_tune(
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        output_dir=str(output_dir),
        num_epochs=args.epochs,
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    final_gleu = metrics.get("eval_gleu", metrics.get("gleu", 0.0))
    LOGGER.info("Final validation GLEU: %.4f", float(final_gleu))
    print(f"Final GLEU: {float(final_gleu):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

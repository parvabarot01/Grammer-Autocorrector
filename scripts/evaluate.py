"""Evaluate a trained T5 grammar correction checkpoint on a test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.models.t5_corrector import T5GrammarCorrector
from src.utils.config import BASE_DIR
from src.utils.evaluation import Evaluator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default=str(BASE_DIR / "models" / "t5"))
    parser.add_argument("--data_dir", default=str(BASE_DIR / "data" / "processed"))
    parser.add_argument(
        "--output_path",
        default=str(BASE_DIR / "results" / "evaluation_report.md"),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=4)
    return parser.parse_args()


def resolve_test_path(data_dir: Path) -> Path:
    """Resolve the test split JSONL path from a processed directory."""

    preferred_prefixes = ["jfleg", "grammar_correction"]
    for prefix in preferred_prefixes:
        candidate = data_dir / f"{prefix}_test.jsonl"
        if candidate.exists():
            return candidate

    candidates = sorted(data_dir.glob("*_test.jsonl"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not find a test split in {data_dir}.")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSON Lines into a list of row dictionaries."""

    with path.open("r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def main() -> int:
    """Evaluate a saved T5 checkpoint and emit a markdown report."""

    args = parse_args()
    data_dir = Path(args.data_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    test_path = resolve_test_path(data_dir)
    dataset = load_jsonl(test_path)
    evaluator = Evaluator()
    corrector = T5GrammarCorrector.from_pretrained(args.model_dir)

    originals = [row["original"] for row in dataset]
    references = [row.get("references", [row["corrected"]]) for row in dataset]
    reference_texts = [values[0] for values in references]
    predictions = corrector.correct_batch(
        originals,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
    )
    gold_edits = [
        evaluator._extract_edits(row["original"], row["corrected"]) for row in dataset
    ]

    metrics = {
        "model_name": Path(args.model_dir).name,
        "gleu": evaluator.compute_gleu(predictions, references),
        "rouge": evaluator.compute_rouge(predictions, reference_texts),
        "m2": evaluator.compute_m2_score(originals, predictions, gold_edits),
        "exact_match": evaluator.compute_exact_match(predictions, reference_texts),
        "samples_evaluated": len(dataset),
    }
    evaluator.generate_evaluation_report(metrics, str(output_path))

    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| GLEU | {metrics['gleu']:.4f} |")
    print(f"| ROUGE-L | {metrics['rouge']['rougeL']:.4f} |")
    print(f"| Exact Match | {metrics['exact_match']:.4f} |")
    print(f"| M2 F0.5 | {metrics['m2']['f05']:.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

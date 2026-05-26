"""Evaluation utilities for grammar correction experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence


@dataclass
class BatchEvaluationResult:
    """Container for normalized evaluation inputs."""

    originals: List[str]
    predictions: List[str]
    references: List[str]
    multi_references: List[List[str]]
    gold_edits: List[List[str]]


class Evaluator:
    """Compute grammar correction metrics and generate reports."""

    def compute_gleu(
        self, predictions: List[str], references: List[List[str]]
    ) -> float:
        """Compute corpus-level GLEU using sacrebleu."""

        self._validate_parallel_lists(
            predictions, references, "predictions", "references"
        )
        if not predictions:
            return 0.0

        try:
            import sacrebleu
        except ImportError as exc:  # pragma: no cover - dependency managed externally
            raise ImportError(
                "sacrebleu is required to compute GLEU. Install it with "
                "`pip install sacrebleu`."
            ) from exc

        transposed_references = [list(group) for group in zip(*references)]
        result = sacrebleu.corpus_gleu(predictions, transposed_references)
        return result.score / 100.0

    def compute_rouge(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute average ROUGE-1, ROUGE-2, and ROUGE-L scores."""

        self._validate_parallel_lists(
            predictions, references, "predictions", "references"
        )
        if not predictions:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        try:
            from rouge_score import rouge_scorer
        except ImportError as exc:  # pragma: no cover - dependency managed externally
            raise ImportError(
                "rouge-score is required to compute ROUGE metrics. Install it with "
                "`pip install rouge-score`."
            ) from exc

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        for prediction, reference in zip(predictions, references):
            scores = scorer.score(reference, prediction)
            for metric_name in totals:
                totals[metric_name] += scores[metric_name].fmeasure

        sample_count = len(predictions)
        return {
            metric_name: score / sample_count for metric_name, score in totals.items()
        }

    def compute_m2_score(
        self, original: List[str], corrected: List[str], gold_edits: List[List[str]]
    ) -> Dict[str, float]:
        """Compute an M2-style Precision, Recall, and F0.5 score."""

        self._validate_parallel_lists(original, corrected, "original", "corrected")
        self._validate_parallel_lists(original, gold_edits, "original", "gold_edits")
        if not original:
            return {"precision": 0.0, "recall": 0.0, "f05": 0.0}

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for source_text, corrected_text, reference_edits in zip(
            original, corrected, gold_edits
        ):
            predicted_edits = {
                self._normalize_edit(edit)
                for edit in self._extract_edits(source_text, corrected_text)
            }
            normalized_gold = {self._normalize_edit(edit) for edit in reference_edits}

            true_positives += len(predicted_edits & normalized_gold)
            false_positives += len(predicted_edits - normalized_gold)
            false_negatives += len(normalized_gold - predicted_edits)

        precision = self._safe_divide(true_positives, true_positives + false_positives)
        recall = self._safe_divide(true_positives, true_positives + false_negatives)
        beta_squared = 0.5**2
        denominator = (beta_squared * precision) + recall
        f05 = (
            ((1 + beta_squared) * precision * recall) / denominator
            if denominator
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f05": f05}

    def compute_exact_match(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Compute case-insensitive exact match accuracy."""

        self._validate_parallel_lists(
            predictions, references, "predictions", "references"
        )
        if not predictions:
            return 0.0

        matches = sum(
            prediction.strip().casefold() == reference.strip().casefold()
            for prediction, reference in zip(predictions, references)
        )
        return matches / len(predictions)

    def evaluate_batch(
        self,
        model_fn: Callable[[List[str]], Sequence[str]],
        dataset: Iterable[Mapping[str, Any]],
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Run evaluation over a dataset and return a combined metrics dictionary."""

        examples = list(dataset)
        normalized = self._normalize_dataset(examples, model_fn, batch_size=batch_size)

        metrics: Dict[str, Any] = {
            "samples_evaluated": len(normalized.predictions),
            "gleu": self.compute_gleu(
                normalized.predictions, normalized.multi_references
            ),
            "rouge": self.compute_rouge(normalized.predictions, normalized.references),
            "exact_match": self.compute_exact_match(
                normalized.predictions, normalized.references
            ),
            "model_name": getattr(model_fn, "__name__", model_fn.__class__.__name__),
        }

        if any(normalized.gold_edits):
            metrics["m2"] = self.compute_m2_score(
                normalized.originals, normalized.predictions, normalized.gold_edits
            )
        else:
            metrics["m2"] = {"precision": None, "recall": None, "f05": None}

        return metrics

    def generate_evaluation_report(
        self, metrics: Dict[str, Any], output_path: str
    ) -> None:
        """Write a markdown evaluation report to disk."""

        timestamp = datetime.now(timezone.utc).isoformat()
        model_name = metrics.get("model_name", "unknown")
        flattened_metrics = self._flatten_metrics(metrics)

        lines = [
            "# Evaluation Report",
            "",
            f"- Generated: {timestamp}",
            f"- Model Name: {model_name}",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]

        for metric_name, value in flattened_metrics.items():
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"| {metric_name} | {formatted_value} |")

        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _normalize_dataset(
        self,
        examples: List[Mapping[str, Any]],
        model_fn: Callable[[List[str]], Sequence[str]],
        batch_size: int,
    ) -> BatchEvaluationResult:
        """Normalize dataset examples into aligned evaluation lists."""

        originals = [self._resolve_original_text(example) for example in examples]
        predictions: List[str] = []

        for batch_start in range(0, len(originals), batch_size):
            batch = originals[batch_start : batch_start + batch_size]
            try:
                batch_predictions = model_fn(batch)
            except TypeError:
                batch_predictions = [model_fn(text) for text in batch]
            if isinstance(batch_predictions, str):
                batch_predictions = [batch_predictions]
            batch_predictions = list(batch_predictions)
            if len(batch_predictions) != len(batch):
                raise ValueError(
                    "model_fn must return one prediction per input text in the batch."
                )
            predictions.extend(batch_predictions)

        references = [self._resolve_reference_text(example) for example in examples]
        multi_references = [
            self._resolve_reference_list(example) for example in examples
        ]
        gold_edits = [list(example.get("gold_edits") or []) for example in examples]

        return BatchEvaluationResult(
            originals=originals,
            predictions=predictions,
            references=references,
            multi_references=multi_references,
            gold_edits=gold_edits,
        )

    def _resolve_original_text(self, example: Mapping[str, Any]) -> str:
        """Resolve the source text field from a dataset record."""

        for key in ("original", "source", "input", "sentence", "text"):
            value = example.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise ValueError(
            "Dataset example must contain one of: original, source, input, "
            "sentence, text."
        )

    def _resolve_reference_text(self, example: Mapping[str, Any]) -> str:
        """Resolve the primary reference text from a dataset record."""

        reference_list = self._resolve_reference_list(example)
        if not reference_list:
            raise ValueError("Dataset example is missing a usable reference text.")
        return reference_list[0]

    def _resolve_reference_list(self, example: Mapping[str, Any]) -> List[str]:
        """Resolve one or more reference texts from a dataset record."""

        for key in ("references", "reference", "targets", "target", "corrections"):
            value = example.get(key)
            if isinstance(value, list) and value:
                return [str(item) for item in value]
            if isinstance(value, str) and value.strip():
                return [value]
        raise ValueError(
            "Dataset example must contain one of: references, reference, targets, "
            "target, corrections."
        )

    def _extract_edits(self, source: str, corrected: str) -> List[str]:
        """Extract lightweight edit representations from source/corrected text."""

        source_tokens = source.split()
        corrected_tokens = corrected.split()
        matcher = SequenceMatcher(a=source_tokens, b=corrected_tokens)
        edits: List[str] = []

        for operation, src_start, src_end, tgt_start, tgt_end in matcher.get_opcodes():
            if operation == "equal":
                continue
            replacement = " ".join(corrected_tokens[tgt_start:tgt_end])
            edits.append(f"{operation}:{src_start}:{src_end}:{replacement}".strip())

        return edits

    def _normalize_edit(self, edit: str) -> str:
        """Normalize a gold edit string into a comparable representation."""

        normalized = edit.strip()
        if normalized.startswith("A "):
            normalized = normalized[2:]
        return normalized.casefold()

    def _flatten_metrics(
        self, metrics: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, Any]:
        """Flatten nested metric dictionaries for markdown reporting."""

        flattened: Dict[str, Any] = {}
        for key, value in metrics.items():
            metric_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(value, dict):
                flattened.update(self._flatten_metrics(value, prefix=metric_key))
            else:
                flattened[metric_key] = value
        return flattened

    def _validate_parallel_lists(
        self,
        left: Sequence[Any],
        right: Sequence[Any],
        left_name: str,
        right_name: str,
    ) -> None:
        """Ensure paired metric inputs share the same length."""

        if len(left) != len(right):
            raise ValueError(
                f"{left_name} and {right_name} must have the same number of items. "
                f"Received {len(left)} and {len(right)}."
            )

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Safely divide two numbers, returning 0.0 for zero denominators."""

        return numerator / denominator if denominator else 0.0

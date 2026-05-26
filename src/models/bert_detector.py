"""BERT-based token-level grammar error detection."""

from __future__ import annotations

import json
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from tqdm.auto import tqdm


LOGGER = logging.getLogger(__name__)


@dataclass
class ErrorSpan:
    """Describe a detected error span."""

    start: int
    end: int
    token: str
    confidence: float
    error_type: str = "UNKNOWN"


class BERTGrammarDetector:
    """Wrap a BERT token classifier for grammar error detection."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        device: str = "auto",
    ) -> None:
        """Initialize the detector with runtime configuration.

        Args:
            model_name: Hugging Face model identifier or local checkpoint path.
            num_labels: Number of token classes. The default maps to CORRECT/ERROR.
            device: Device preference. Use `"auto"` to prefer CUDA.
        """

        self.model_name = model_name
        self.num_labels = num_labels
        self.device = self._resolve_device(device)
        self.max_length = 128
        self.batch_size = 32
        self.learning_rate = 3e-4
        self.confidence_threshold = 0.5
        self.model: Any = None
        self.tokenizer: Any = None

    def load_model(self) -> None:
        """Load the BERT token classifier and tokenizer."""

        transformers = self._import_transformers()
        self.tokenizer = transformers["BertTokenizerFast"].from_pretrained(
            self.model_name
        )
        self.model = transformers["BertForTokenClassification"].from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )
        self._move_model_to_device()
        LOGGER.info(
            "Loaded BERT detector '%s' on %s with %s parameters.",
            self.model_name,
            self.device,
            f"{self._parameter_count():,}",
        )

    def detect_errors(self, text: str) -> List[ErrorSpan]:
        """Detect token-level grammar issues in a single string.

        Args:
            text: Input sentence to analyze.

        Returns:
            List[ErrorSpan]: Detected error spans above the confidence threshold.
        """

        if self.model is None or self.tokenizer is None:
            self.load_model()

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        return self._predict_from_encoded(text, encoded)[0]

    def detect_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[ErrorSpan]]:
        """Detect grammar issues for a batch of texts.

        Args:
            texts: Input texts to analyze.
            batch_size: Number of items to process per inference step.

        Returns:
            List[List[ErrorSpan]]: Detected spans grouped by input text.
        """

        if self.model is None or self.tokenizer is None:
            self.load_model()

        predictions: List[List[ErrorSpan]] = []
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc="Detecting errors",
            leave=False,
        ):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            predictions.extend(self._predict_from_encoded(batch, encoded))
        return predictions

    def has_errors(self, text: str) -> bool:
        """Check whether any error spans are detected in the text.

        Args:
            text: Input sentence to analyze.

        Returns:
            bool: `True` if at least one error token is predicted.
        """

        return bool(self.detect_errors(text))

    def get_error_positions(self, text: str) -> List[int]:
        """Return character positions covered by detected error spans.

        Args:
            text: Input sentence to analyze.

        Returns:
            List[int]: Sorted unique character indices associated with errors.
        """

        positions = {
            position
            for span in self.detect_errors(text)
            for position in range(span.start, span.end)
        }
        return sorted(positions)

    def fine_tune(
        self,
        train_dataset: Iterable[Mapping[str, Any]],
        val_dataset: Iterable[Mapping[str, Any]],
        output_dir: str,
        num_epochs: int = 3,
    ) -> Dict[str, Any]:
        """Fine-tune the BERT detector on labeled token-classification data.

        Args:
            train_dataset: Training dataset with token-level `labels`.
            val_dataset: Validation dataset with token-level `labels`.
            output_dir: Directory for checkpoints and logs.
            num_epochs: Number of fine-tuning epochs.

        Returns:
            Dict[str, Any]: Combined training and validation metrics.
        """

        if self.model is None or self.tokenizer is None:
            self.load_model()

        torch = self._import_torch()
        transformers = self._import_transformers()
        datasets_lib = self._import_datasets()
        sklearn_metrics = self._import_sklearn_metrics()

        train_hf = self._as_hf_dataset(train_dataset, datasets_lib)
        val_hf = self._as_hf_dataset(val_dataset, datasets_lib)
        class_weights = self._compute_class_weights(train_hf, torch).to(self.device)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        class WeightedTokenTrainer(transformers["Trainer"]):
            """Trainer subclass using weighted cross entropy."""

            def compute_loss(
                self,
                model: Any,
                inputs: Dict[str, Any],
                return_outputs: bool = False,
            ) -> Any:
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=class_weights,
                    ignore_index=-100,
                )
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
                inputs["labels"] = labels
                return (loss, outputs) if return_outputs else loss

        def compute_metrics(eval_prediction: Any) -> Dict[str, float]:
            predictions = eval_prediction.predictions
            labels = eval_prediction.label_ids
            if isinstance(predictions, tuple):
                predictions = predictions[0]

            predicted_ids = predictions.argmax(axis=-1)
            filtered_predictions: List[int] = []
            filtered_labels: List[int] = []
            for prediction_row, label_row in zip(predicted_ids, labels):
                for predicted_label, gold_label in zip(prediction_row, label_row):
                    if int(gold_label) == -100:
                        continue
                    filtered_predictions.append(int(predicted_label))
                    filtered_labels.append(int(gold_label))

            precision, recall, f1, _ = sklearn_metrics(
                filtered_labels,
                filtered_predictions,
                average="binary",
                zero_division=0,
            )
            return {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

        training_args = transformers["TrainingArguments"](
            output_dir=str(output_path),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=[],
        )

        trainer = WeightedTokenTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            tokenizer=self.tokenizer,
            data_collator=transformers["default_data_collator"],
            callbacks=[
                transformers["EarlyStoppingCallback"](early_stopping_patience=2)
            ],
            compute_metrics=compute_metrics,
        )

        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        trainer.save_model(str(output_path))
        self.save(str(output_path))

        metrics = dict(train_result.metrics)
        metrics.update(eval_metrics)
        metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint
        return metrics

    def save(self, output_dir: str) -> None:
        """Save the detector checkpoint and local metadata.

        Args:
            output_dir: Destination directory for checkpoint files.
        """

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Cannot save before the model and tokenizer are loaded.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        metadata = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "confidence_threshold": self.confidence_threshold,
        }
        (output_path / "bert_detector_config.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "BERTGrammarDetector":
        """Load a detector from a local checkpoint directory.

        Args:
            model_dir: Local directory containing saved detector assets.

        Returns:
            BERTGrammarDetector: Instantiated detector with loaded weights.
        """

        metadata_path = Path(model_dir) / "bert_detector_config.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        instance = cls(
            model_name=model_dir,
            num_labels=int(metadata.get("num_labels", 2)),
            device=str(metadata.get("device", "auto")),
        )
        instance.max_length = int(metadata.get("max_length", instance.max_length))
        instance.batch_size = int(metadata.get("batch_size", instance.batch_size))
        instance.learning_rate = float(
            metadata.get("learning_rate", instance.learning_rate)
        )
        instance.confidence_threshold = float(
            metadata.get("confidence_threshold", instance.confidence_threshold)
        )

        transformers = instance._import_transformers()
        instance.tokenizer = transformers["BertTokenizerFast"].from_pretrained(
            model_dir
        )
        instance.model = transformers["BertForTokenClassification"].from_pretrained(
            model_dir
        )
        instance._move_model_to_device()
        return instance

    def _predict_from_encoded(
        self, texts: Any, encoded: MutableMapping[str, Any]
    ) -> List[List[ErrorSpan]]:
        """Run the model on encoded inputs and decode span predictions."""

        if isinstance(texts, str):
            text_list = [texts]
        else:
            text_list = list(texts)

        offsets = self._pop_offsets(encoded)
        model_inputs = self._move_batch_to_device(encoded)

        with self._inference_context():
            outputs = self.model(**model_inputs)

        logits = self._to_nested_list(outputs.logits)
        input_ids = self._to_nested_list(encoded["input_ids"])

        predictions: List[List[ErrorSpan]] = []
        for batch_index, text in enumerate(text_list):
            token_ids = input_ids[batch_index]
            token_offsets = offsets[batch_index]
            token_strings = self.tokenizer.convert_ids_to_tokens(token_ids)
            spans: List[ErrorSpan] = []

            for token_index, token_logits in enumerate(logits[batch_index]):
                if token_index >= len(token_offsets):
                    break
                start, end = token_offsets[token_index]
                if start == end:
                    continue

                label_id, confidence = self._predict_label(token_logits)
                if label_id != 1 or confidence < self.confidence_threshold:
                    continue

                token_text = (
                    text[start:end]
                    if end <= len(text)
                    else token_strings[token_index]
                )
                spans.append(
                    ErrorSpan(
                        start=int(start),
                        end=int(end),
                        token=token_text,
                        confidence=float(confidence),
                        error_type="GRAMMAR_ERROR",
                    )
                )
            predictions.append(spans)

        return predictions

    def _compute_class_weights(self, dataset: Any, torch: Any) -> Any:
        """Compute class weights for imbalanced token labels."""

        correct_count = 0
        error_count = 0
        for labels in dataset["labels"]:
            for label in labels:
                if int(label) == -100:
                    continue
                if int(label) == 1:
                    error_count += 1
                else:
                    correct_count += 1

        total = max(correct_count + error_count, 1)
        weight_correct = total / max(correct_count * 2, 1)
        weight_error = total / max(error_count * 2, 1)
        return torch.tensor([weight_correct, weight_error], dtype=torch.float32)

    def _resolve_device(self, device: str) -> str:
        """Resolve the runtime device string."""

        if device != "auto":
            return device
        torch = self._maybe_import_torch()
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _parameter_count(self) -> int:
        """Return the number of model parameters."""

        if self.model is None:
            return 0
        return sum(int(parameter.numel()) for parameter in self.model.parameters())

    def _move_model_to_device(self) -> None:
        """Move the underlying model to the configured device."""

        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(self.device)

    def _move_batch_to_device(
        self, batch: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Move tensor-like batch values onto the configured device."""

        moved: MutableMapping[str, Any] = type(batch)()
        for key, value in batch.items():
            moved[key] = value.to(self.device) if hasattr(value, "to") else value
        return moved

    def _pop_offsets(self, encoded: MutableMapping[str, Any]) -> List[List[List[int]]]:
        """Extract offset mappings from tokenizer output."""

        offsets = encoded.pop("offset_mapping")
        return self._to_nested_list(offsets)

    def _predict_label(self, token_logits: Any) -> tuple[int, float]:
        """Convert per-token logits into a class label and confidence."""

        probabilities = self._softmax(self._to_list(token_logits))
        label_id = max(range(len(probabilities)), key=probabilities.__getitem__)
        return label_id, probabilities[label_id]

    def _softmax(self, values: List[float]) -> List[float]:
        """Compute a numerically stable softmax over a list of values."""

        max_value = max(values)
        exponentials = [math.exp(value - max_value) for value in values]
        denominator = sum(exponentials) or 1.0
        return [value / denominator for value in exponentials]

    def _to_nested_list(self, value: Any) -> List[Any]:
        """Convert tensor-like data into a nested Python list."""

        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    def _to_list(self, value: Any) -> List[float]:
        """Convert tensor-like or tuple data into a flat list."""

        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            return list(value.tolist())
        return list(value)

    def _inference_context(self) -> Any:
        """Return a no-grad context when torch is available."""

        torch = self._maybe_import_torch()
        if torch is None:
            return nullcontext()
        return torch.no_grad()

    def _as_hf_dataset(
        self, dataset: Iterable[Mapping[str, Any]], datasets_lib: Any
    ) -> Any:
        """Normalize a token-label dataset into a Hugging Face dataset."""

        if hasattr(dataset, "map") and hasattr(dataset, "column_names"):
            return dataset
        return datasets_lib["Dataset"].from_list(list(dataset))

    def _import_transformers(self) -> Dict[str, Any]:
        """Import transformer classes lazily."""

        try:
            from transformers import (
                BertForTokenClassification,
                BertTokenizerFast,
                EarlyStoppingCallback,
                Trainer,
                TrainingArguments,
                default_data_collator,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is required for BERTGrammarDetector. "
                "Install it with `pip install transformers`."
            ) from exc

        return {
            "BertForTokenClassification": BertForTokenClassification,
            "BertTokenizerFast": BertTokenizerFast,
            "EarlyStoppingCallback": EarlyStoppingCallback,
            "Trainer": Trainer,
            "TrainingArguments": TrainingArguments,
            "default_data_collator": default_data_collator,
        }

    def _import_datasets(self) -> Dict[str, Any]:
        """Import Hugging Face datasets lazily."""

        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for fine-tuning. "
                "Install it with `pip install datasets`."
            ) from exc
        return {"Dataset": Dataset}

    def _import_sklearn_metrics(self) -> Any:
        """Import sklearn metrics lazily."""

        try:
            from sklearn.metrics import precision_recall_fscore_support
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for BERT metric computation. "
                "Install it with `pip install scikit-learn`."
            ) from exc
        return precision_recall_fscore_support

    def _maybe_import_torch(self) -> Optional[Any]:
        """Import torch if it is available."""

        try:
            import torch
        except ImportError:
            return None
        return torch

    def _import_torch(self) -> Any:
        """Import torch or raise a helpful error."""

        torch = self._maybe_import_torch()
        if torch is None:
            raise ImportError(
                "torch is required for BERTGrammarDetector. "
                "Install it with `pip install torch`."
            )
        return torch

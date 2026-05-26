"""T5-based grammar correction model wrapper."""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np
from tqdm.auto import tqdm

from src.utils.evaluation import Evaluator
from src.utils.preprocessing import GrammarPreprocessor


LOGGER = logging.getLogger(__name__)


class T5GrammarCorrector:
    """Wrap a T5 model for grammar correction tasks."""

    def __init__(self, model_name: str = "t5-base", device: str = "auto") -> None:
        """Initialize the corrector with lightweight configuration.

        Args:
            model_name: Hugging Face model identifier or local checkpoint path.
            device: Device preference. Use `"auto"` to prefer CUDA when available.
        """

        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model: Any = None
        self.tokenizer: Any = None
        self.max_length = 128
        self.batch_size = 16
        self.learning_rate = 3e-4
        self.num_beams = 4
        self.preprocessor = GrammarPreprocessor()
        self.evaluator = Evaluator()

    def load_model(self) -> None:
        """Load the T5 model and tokenizer onto the configured device.

        Raises:
            ImportError: If the required `transformers` dependency is unavailable.
        """

        transformers = self._import_transformers()
        self.tokenizer = transformers["T5Tokenizer"].from_pretrained(self.model_name)
        self.model = transformers["T5ForConditionalGeneration"].from_pretrained(
            self.model_name
        )
        self._move_model_to_device()
        LOGGER.info(
            "Loaded T5 model '%s' on %s with %s parameters.",
            self.model_name,
            self.device,
            f"{self._parameter_count():,}",
        )

    def preprocess(self, texts: List[str]) -> Any:
        """Tokenize a batch of input texts for grammar correction.

        Args:
            texts: Raw text inputs to correct.

        Returns:
            Any: Tokenizer output compatible with Hugging Face generation APIs.

        Raises:
            RuntimeError: If the tokenizer has not been loaded yet.
        """

        self._ensure_tokenizer()
        cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
        prefixed_texts = [f"grammar: {text}" for text in cleaned_texts]
        return self.tokenizer(
            prefixed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def correct(
        self, text: str, num_beams: int = 4, max_length: int = 128
    ) -> str:
        """Correct a single sentence with beam search decoding.

        Args:
            text: Input sentence to correct.
            num_beams: Beam width for generation.
            max_length: Maximum decoded sequence length.

        Returns:
            str: Corrected output sentence.
        """

        if self.model is None or self.tokenizer is None:
            self.load_model()

        encoded = self.preprocess([text])
        model_inputs = self._move_batch_to_device(encoded)
        generation_kwargs = {"num_beams": num_beams, "max_length": max_length}

        with self._inference_context():
            generated = self.model.generate(**model_inputs, **generation_kwargs)

        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    def correct_batch(
        self, texts: List[str], batch_size: int = 16, num_beams: int = 4
    ) -> List[str]:
        """Correct a list of sentences in mini-batches.

        Args:
            texts: Input sentences to correct.
            batch_size: Number of examples to process per generation step.
            num_beams: Beam width for generation.

        Returns:
            List[str]: Corrected outputs aligned with the input ordering.
        """

        if self.model is None or self.tokenizer is None:
            self.load_model()

        predictions: List[str] = []
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc="Correcting batches",
            leave=False,
        ):
            batch = texts[start : start + batch_size]
            encoded = self.preprocess(batch)
            model_inputs = self._move_batch_to_device(encoded)
            generation_kwargs = {
                "num_beams": num_beams,
                "max_length": self.max_length,
            }

            with self._inference_context():
                generated = self.model.generate(**model_inputs, **generation_kwargs)

            predictions.extend(
                self.tokenizer.decode(sequence, skip_special_tokens=True).strip()
                for sequence in generated
            )

        return predictions

    def fine_tune(
        self,
        train_dataset: Iterable[Mapping[str, Any]],
        val_dataset: Iterable[Mapping[str, Any]],
        output_dir: str,
        num_epochs: int = 5,
    ) -> Dict[str, Any]:
        """Fine-tune the T5 model on a correction dataset.

        Args:
            train_dataset: Training examples containing source and target text.
            val_dataset: Validation examples containing source and target text.
            output_dir: Directory for checkpoints and trainer outputs.
            num_epochs: Number of training epochs.

        Returns:
            Dict[str, Any]: Combined training and validation metrics.
        """

        if self.model is None or self.tokenizer is None:
            self.load_model()

        datasets_lib = self._import_datasets()
        transformers = self._import_transformers()

        train_hf = self._as_hf_dataset(train_dataset, datasets_lib)
        val_hf = self._as_hf_dataset(val_dataset, datasets_lib)

        train_tokenized = train_hf.map(
            self._tokenize_training_batch,
            batched=True,
            remove_columns=train_hf.column_names,
        )
        val_tokenized = val_hf.map(
            self._tokenize_training_batch,
            batched=True,
            remove_columns=val_hf.column_names,
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        data_collator = transformers["DataCollatorForSeq2Seq"](
            tokenizer=self.tokenizer, model=self.model
        )

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
            metric_for_best_model="gleu",
            greater_is_better=True,
            report_to=[],
            predict_with_generate=True,
        )

        trainer = transformers["Trainer"](
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._build_compute_metrics(),
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
        """Save model weights, tokenizer, and local metadata.

        Args:
            output_dir: Destination directory for the checkpoint assets.
        """

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Cannot save before the model and tokenizer are loaded.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        metadata = {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_beams": self.num_beams,
        }
        (output_path / "t5_corrector_config.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "T5GrammarCorrector":
        """Load a corrector from a local checkpoint directory.

        Args:
            model_dir: Local directory containing saved model assets.

        Returns:
            T5GrammarCorrector: Instantiated corrector with model and tokenizer.
        """

        instance = cls(model_name=model_dir)
        transformers = instance._import_transformers()
        metadata_path = Path(model_dir) / "t5_corrector_config.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            instance.max_length = int(metadata.get("max_length", instance.max_length))
            instance.batch_size = int(metadata.get("batch_size", instance.batch_size))
            instance.learning_rate = float(
                metadata.get("learning_rate", instance.learning_rate)
            )
            instance.num_beams = int(metadata.get("num_beams", instance.num_beams))

        instance.tokenizer = transformers["T5Tokenizer"].from_pretrained(model_dir)
        instance.model = transformers["T5ForConditionalGeneration"].from_pretrained(
            model_dir
        )
        instance._move_model_to_device()
        return instance

    def _resolve_device(self, device: str) -> str:
        """Resolve the target runtime device."""

        torch = self._maybe_import_torch()
        if device != "auto":
            return device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _parameter_count(self) -> int:
        """Count model parameters for logging purposes."""

        if self.model is None:
            return 0
        return sum(int(parameter.numel()) for parameter in self.model.parameters())

    def _move_model_to_device(self) -> None:
        """Move the underlying model to the configured device when supported."""

        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(self.device)

    def _ensure_tokenizer(self) -> None:
        """Validate that a tokenizer has been loaded."""

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded. Call load_model() first.")

    def _move_batch_to_device(
        self, batch: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Move tensor-like batch values onto the configured device."""

        moved: MutableMapping[str, Any] = type(batch)()
        for key, value in batch.items():
            moved[key] = value.to(self.device) if hasattr(value, "to") else value
        return moved

    def _inference_context(self) -> Any:
        """Return a no-grad context when torch is available."""

        torch = self._maybe_import_torch()
        if torch is None:
            return nullcontext()
        return torch.no_grad()

    def _build_compute_metrics(self) -> Any:
        """Create the Hugging Face metrics callback used during training."""

        def compute_metrics(eval_prediction: Any) -> Dict[str, float]:
            predictions = eval_prediction.predictions
            labels = eval_prediction.label_ids
            if isinstance(predictions, tuple):
                predictions = predictions[0]

            pad_token_id = self.tokenizer.pad_token_id
            labels = np.where(labels == -100, pad_token_id, labels)
            decoded_predictions = self.tokenizer.batch_decode(
                predictions,
                skip_special_tokens=True,
            )
            decoded_labels = self.tokenizer.batch_decode(
                labels,
                skip_special_tokens=True,
            )
            gleu = self.evaluator.compute_gleu(
                [prediction.strip() for prediction in decoded_predictions],
                [[label.strip()] for label in decoded_labels],
            )
            return {"gleu": float(gleu)}

        return compute_metrics

    def _tokenize_training_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Tokenize a dataset batch for sequence-to-sequence fine-tuning."""

        rows = self._rows_from_batch(batch)
        sources = [self._extract_source(record) for record in rows]
        targets = [self._extract_target(record) for record in rows]
        model_inputs = self.tokenizer(
            [f"grammar: {self.preprocessor.clean_text(text)}" for text in sources],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        labels = self.tokenizer(
            text_target=[self.preprocessor.clean_text(text) for text in targets],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _rows_from_batch(self, batch: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Convert a dictionary-of-lists batch into row dictionaries."""

        keys = list(batch.keys())
        return [
            {key: batch[key][index] for key in keys}
            for index in range(len(batch[keys[0]]))
        ]

    def _extract_source(self, row: Mapping[str, Any]) -> str:
        """Extract the source sentence from a dataset row."""

        for key in ("original", "source", "input", "sentence", "text"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise ValueError("Training row does not include a usable source sentence.")

    def _extract_target(self, row: Mapping[str, Any]) -> str:
        """Extract the correction target from a dataset row."""

        for key in ("corrected", "target", "reference"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value
        references = row.get("references")
        if isinstance(references, list) and references:
            return str(references[0])
        raise ValueError("Training row does not include a usable target sentence.")

    def _as_hf_dataset(
        self, dataset: Iterable[Mapping[str, Any]], datasets_lib: Any
    ) -> Any:
        """Normalize user-provided training data into a Hugging Face dataset."""

        if hasattr(dataset, "map") and hasattr(dataset, "column_names"):
            return dataset
        return datasets_lib["Dataset"].from_list(list(dataset))

    def _import_transformers(self) -> Dict[str, Any]:
        """Import transformer classes lazily."""

        try:
            from transformers import (
                DataCollatorForSeq2Seq,
                T5ForConditionalGeneration,
                T5Tokenizer,
                Trainer,
                TrainingArguments,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is required for T5GrammarCorrector. "
                "Install it with `pip install transformers`."
            ) from exc

        return {
            "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
            "T5ForConditionalGeneration": T5ForConditionalGeneration,
            "T5Tokenizer": T5Tokenizer,
            "Trainer": Trainer,
            "TrainingArguments": TrainingArguments,
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

    def _maybe_import_torch(self) -> Optional[Any]:
        """Import torch if it is available in the current environment."""

        try:
            import torch
        except ImportError:
            return None
        return torch

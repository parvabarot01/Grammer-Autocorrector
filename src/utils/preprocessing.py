"""Text preprocessing utilities for grammar correction workflows."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Tuple


class GrammarPreprocessor:
    """Utility class for cleaning, tokenizing, and validating text."""

    _ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
    _MULTISPACE_PATTERN = re.compile(r"\s+")
    _ABBREVIATIONS = {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "st.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
        "u.s.",
        "u.k.",
    }
    _DOT_SENTINEL = "<DOT>"

    def clean_text(self, text: str) -> str:
        """Normalize and clean user-provided text."""

        normalized = unicodedata.normalize("NFKC", text)
        without_zero_width = self._ZERO_WIDTH_PATTERN.sub("", normalized)
        collapsed = self._MULTISPACE_PATTERN.sub(" ", without_zero_width)
        return collapsed.strip()

    def tokenize_for_t5(
        self, text: str, tokenizer: Any, max_length: int
    ) -> Dict[str, Any]:
        """Tokenize text for T5 sequence-to-sequence correction."""

        cleaned_text = self.clean_text(text)
        prefixed_text = f"grammar: {cleaned_text}"
        return tokenizer(
            prefixed_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def tokenize_for_bert(
        self, text: str, tokenizer: Any, max_length: int
    ) -> Dict[str, Any]:
        """Tokenize text for BERT token classification."""

        cleaned_text = self.clean_text(text)
        encoded = tokenizer(
            cleaned_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if "token_type_ids" not in encoded:
            encoded["token_type_ids"] = self._build_zero_like_structure(
                encoded.get("input_ids", [])
            )
        return encoded

    def split_into_sentences(self, text: str) -> List[str]:
        """Split a block of text into sentences using simple boundary rules."""

        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []

        protected_text = cleaned_text
        for abbreviation in sorted(self._ABBREVIATIONS, key=len, reverse=True):
            protected_text = re.sub(
                re.escape(abbreviation),
                lambda match: match.group(0).replace(".", self._DOT_SENTINEL),
                protected_text,
                flags=re.IGNORECASE,
            )

        parts = re.split(r"(?<=[.!?])\s+", protected_text)
        return [
            part.replace(self._DOT_SENTINEL, ".").strip()
            for part in parts
            if part.strip()
        ]

    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts and remove empty entries."""

        cleaned_batch = [self.clean_text(text) for text in texts]
        return [text for text in cleaned_batch if text]

    def validate_input(self, text: str, max_length: int = 1000) -> Tuple[bool, str]:
        """Validate user input before it enters the correction pipeline."""

        if text is None:
            return False, "Input text cannot be None."

        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return False, "Input text cannot be empty."

        if len(cleaned_text) > max_length:
            return False, f"Input text exceeds the maximum length of {max_length}."

        if any(
            not character.isprintable() and character not in "\n\r\t"
            for character in text
        ):
            return False, "Input text contains non-printable characters."

        return True, ""

    def _build_zero_like_structure(self, value: Any) -> Any:
        """Build a zero-filled structure mirroring the tokenizer output shape."""

        try:
            import numpy as np  # type: ignore
        except ImportError:  # pragma: no cover
            np = None

        try:
            import torch  # type: ignore
        except ImportError:  # pragma: no cover
            torch = None

        if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(value):
            return torch.zeros_like(value)
        if np is not None and isinstance(value, np.ndarray):
            return np.zeros_like(value)
        if isinstance(value, list):
            return [self._build_zero_like_structure(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._build_zero_like_structure(item) for item in value)
        if isinstance(value, int):
            return 0
        return value

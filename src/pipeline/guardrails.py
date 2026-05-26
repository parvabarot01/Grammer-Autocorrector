"""Responsible AI guardrails for the grammar correction pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class GuardrailResult:
    """Represent the result of a validation stage."""

    passed: bool
    violations: List[str]
    sanitized_text: str
    severity: str


@dataclass
class ToxicityResult:
    """Represent a toxicity scan result."""

    score: float
    is_toxic: bool
    detected_terms: List[str]


@dataclass
class BiasResult:
    """Represent a bias scan result."""

    has_bias: bool
    bias_types: List[str]
    suggestions: List[str]


@dataclass
class FullGuardrailReport:
    """Aggregate all guardrail checks for a request/response pair."""

    input_valid: GuardrailResult
    toxicity: ToxicityResult
    bias: BiasResult
    output_valid: Optional[GuardrailResult]
    overall_passed: bool
    timestamp: str


class GuardrailViolation(Exception):
    """Structured guardrail exception."""

    def __init__(self, violation_type: str, message: str, severity: str) -> None:
        super().__init__(message)
        self.violation_type = violation_type
        self.severity = severity


class GrammarGuardrails:
    """Validate and sanitize inputs and outputs for grammar correction."""

    PROMPT_INJECTION_PATTERNS = [
        r"ignore\s+previous",
        r"disregard\s+instructions",
        r"system\s+prompt",
        r"forget\s+all\s+previous",
        r"developer\s+message",
        r"override\s+safety",
    ]

    TOXIC_TERMS = [
        "idiot",
        "stupid",
        "moron",
        "dumb",
        "trash",
        "garbage",
        "loser",
        "hate",
        "worthless",
        "ugly",
        "fat",
        "lazy",
        "pathetic",
        "jerk",
        "fool",
        "shut up",
        "kill",
        "die",
        "racist",
        "sexist",
        "bigot",
        "creep",
        "pervert",
        "scum",
        "nasty",
        "filthy",
        "retard",
        "psycho",
        "lunatic",
        "crazy",
        "pig",
        "slob",
        "bastard",
        "slur",
        "violent",
        "toxic",
        "abuse",
        "harass",
        "attack",
        "destroy",
        "punish",
        "threat",
        "terror",
        "bully",
        "disgusting",
        "savage",
        "evil",
        "vile",
        "repulsive",
        "freak",
    ]

    BIAS_PATTERNS = {
        "gendered_job_title": {
            "pattern": r"\b(chairman|fireman|policeman|stewardess)\b",
            "suggestion": "Use a gender-neutral job title where possible.",
        },
        "gender_assumption": {
            "pattern": r"\b(he|she)\b\s+(must|should|will)\b",
            "suggestion": (
                "Avoid unnecessary gender assumptions in examples or instructions."
            ),
        },
        "male_default": {
            "pattern": r"\bmanpower\b|\bguys\b",
            "suggestion": (
                "Consider more inclusive wording such as 'workforce' or 'team'."
            ),
        },
    }

    def __init__(
        self,
        max_input_length: int = 1000,
        max_output_length: int = 1200,
        toxicity_threshold: float = 0.8,
        enable_bias_check: bool = True,
    ) -> None:
        """Configure guardrail thresholds and behavior."""

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.toxicity_threshold = toxicity_threshold
        self.enable_bias_check = enable_bias_check

    def validate_input(self, text: str) -> GuardrailResult:
        """Validate raw user input before correction.

        Args:
            text: User-provided text.

        Returns:
            GuardrailResult: Validation status and sanitized text.
        """

        normalized = self._sanitize_raw(text or "", truncate=False)
        sanitized = normalized[: self.max_input_length]
        violations: List[str] = []
        severity = "none"
        passed = True

        if not normalized.strip():
            return GuardrailResult(
                passed=False,
                violations=["Input text cannot be empty."],
                sanitized_text="",
                severity="error",
            )

        if len(normalized) > self.max_input_length:
            return GuardrailResult(
                passed=False,
                violations=[
                    (
                        "Input exceeds the maximum length of "
                        f"{self.max_input_length} characters."
                    )
                ],
                sanitized_text=sanitized,
                severity="error",
            )

        if len(normalized) > 500:
            violations.append(
                "Input length exceeds the warning threshold of 500 characters."
            )
            severity = "warning"

        if any(
            not character.isprintable() and character not in "\n\r\t"
            for character in (text or "")
        ):
            violations.append(
                "Non-printable characters were removed during sanitization."
            )
            severity = "warning" if severity == "none" else severity

        if self._detect_prompt_injection(text or ""):
            return GuardrailResult(
                passed=False,
                violations=["Prompt injection pattern detected in the input."],
                sanitized_text=sanitized,
                severity="error",
            )

        return GuardrailResult(
            passed=passed,
            violations=violations,
            sanitized_text=sanitized,
            severity=severity,
        )

    def validate_output(self, original: str, corrected: str) -> GuardrailResult:
        """Validate model output before returning it to the caller."""

        normalized = self._sanitize_raw(corrected or "", truncate=False)
        sanitized = normalized[: self.max_output_length]
        if not normalized.strip():
            return GuardrailResult(
                passed=False,
                violations=["Corrected output cannot be empty."],
                sanitized_text="",
                severity="error",
            )

        violations: List[str] = []
        severity = "none"
        passed = True

        original_length = max(len(original.strip()), 1)
        output_ratio = len(normalized.strip()) / original_length
        if output_ratio > 2.0 or len(normalized) > self.max_output_length:
            return GuardrailResult(
                passed=False,
                violations=["Output is disproportionately longer than the input."],
                sanitized_text=sanitized,
                severity="error",
            )

        if not self._preserves_core_meaning(original, normalized):
            violations.append("Output may not preserve the core meaning of the input.")
            severity = "warning"

        if self._contains_unknown_tokens(normalized):
            return GuardrailResult(
                passed=False,
                violations=["Output introduces unknown or placeholder tokens."],
                sanitized_text=sanitized,
                severity="error",
            )

        return GuardrailResult(
            passed=passed,
            violations=violations,
            sanitized_text=sanitized,
            severity=severity,
        )

    def check_toxicity(self, text: str) -> ToxicityResult:
        """Run a keyword-based toxicity scan."""

        lowered = text.casefold()
        detected_terms = [
            term
            for term in self.TOXIC_TERMS
            if re.search(rf"\b{re.escape(term)}\b", lowered)
        ]
        score = min(1.0, len(detected_terms) / 5.0)
        return ToxicityResult(
            score=score,
            is_toxic=score >= self.toxicity_threshold or len(detected_terms) >= 4,
            detected_terms=detected_terms,
        )

    def check_bias(self, text: str) -> BiasResult:
        """Scan text for simple gender-coded or exclusionary language patterns."""

        if not self.enable_bias_check:
            return BiasResult(has_bias=False, bias_types=[], suggestions=[])

        lowered = text.casefold()
        bias_types: List[str] = []
        suggestions: List[str] = []
        for bias_type, config in self.BIAS_PATTERNS.items():
            if re.search(config["pattern"], lowered):
                bias_types.append(bias_type)
                suggestions.append(config["suggestion"])

        return BiasResult(
            has_bias=bool(bias_types),
            bias_types=bias_types,
            suggestions=suggestions,
        )

    def sanitize(self, text: str) -> str:
        """Sanitize text by removing control characters and dangerous patterns."""

        return self._sanitize_raw(text, truncate=True)

    def _sanitize_raw(self, text: str, truncate: bool) -> str:
        """Internal sanitizer with optional truncation control."""

        cleaned = "".join(
            character
            for character in text
            if character.isprintable() or character in "\n\r\t"
        )
        cleaned = re.sub(r"[\x00-\x1f\x7f]", "", cleaned)
        for pattern in self.PROMPT_INJECTION_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if truncate:
            return cleaned[: self.max_input_length]
        return cleaned

    def run_all_checks(
        self, input_text: str, output_text: str = None
    ) -> FullGuardrailReport:
        """Run all guardrail checks for the request lifecycle."""

        input_valid = self.validate_input(input_text)
        toxicity = self.check_toxicity(input_valid.sanitized_text)
        bias = self.check_bias(input_valid.sanitized_text)
        output_valid = (
            self.validate_output(input_valid.sanitized_text, output_text)
            if output_text is not None
            else None
        )
        overall_passed = (
            input_valid.passed
            and not toxicity.is_toxic
            and (output_valid.passed if output_valid is not None else True)
        )
        return FullGuardrailReport(
            input_valid=input_valid,
            toxicity=toxicity,
            bias=bias,
            output_valid=output_valid,
            overall_passed=overall_passed,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _detect_prompt_injection(self, text: str) -> bool:
        """Detect known prompt-injection patterns."""

        return any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in self.PROMPT_INJECTION_PATTERNS
        )

    def _preserves_core_meaning(self, original: str, corrected: str) -> bool:
        """Estimate whether output preserves the original core meaning."""

        original_keywords = {
            word
            for word in re.findall(r"[A-Za-z']+", original.casefold())
            if len(word) > 2
        }
        corrected_keywords = {
            word
            for word in re.findall(r"[A-Za-z']+", corrected.casefold())
            if len(word) > 2
        }
        if not original_keywords:
            return True
        overlap = len(original_keywords & corrected_keywords) / len(original_keywords)
        return overlap >= 0.4

    def _contains_unknown_tokens(self, text: str) -> bool:
        """Detect placeholder or obviously broken output tokens."""

        return bool(re.search(r"<unk>|\[unk\]|\?{3,}", text, flags=re.IGNORECASE))

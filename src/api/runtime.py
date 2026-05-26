"""Runtime service container for the FastAPI grammar correction API."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

from src.models import BERTGrammarDetector, ErrorSpan, T5GrammarCorrector
from src.pipeline import (
    FullGuardrailReport,
    GrammarGuardrails,
    GrammarRAGPipeline,
    GuardrailViolation,
    PromptVersionManager,
)
from src.utils.config import Config
from src.utils.evaluation import Evaluator

LOGGER = logging.getLogger(__name__)


class APIRuntime:
    """Coordinate API-facing model, retrieval, and guardrail operations."""

    def __init__(self, config: Config) -> None:
        """Initialize runtime services from application configuration.

        Args:
            config: Fully-populated project configuration.
        """

        self.config = config
        self.t5 = T5GrammarCorrector(config.model.t5_model_name)
        self.bert = BERTGrammarDetector(config.model.bert_model_name)
        self.rag = GrammarRAGPipeline(
            embedding_model=config.rag.embedding_model_name,
            vector_store_path=str(config.rag.vector_store_path),
            top_k=config.rag.top_k,
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
        )
        self.guardrails = GrammarGuardrails(
            max_input_length=config.guardrails.max_input_length,
            max_output_length=max(int(config.guardrails.max_input_length * 1.2), 1200),
            toxicity_threshold=config.guardrails.toxicity_threshold,
            enable_bias_check=True,
        )
        self.prompt_manager = PromptVersionManager()
        self.evaluator = Evaluator()
        self.models_loaded = False
        self.load_errors: List[str] = []

    def initialize(self) -> None:
        """Load runtime dependencies and initialize the knowledge base."""

        self.load_errors = []
        self._ensure_knowledge_base()
        t5_loaded = self._try_load_model("t5", self.t5.load_model)
        bert_loaded = self._try_load_model("bert", self.bert.load_model)
        self.models_loaded = t5_loaded and bert_loaded
        LOGGER.info(
            "API runtime initialized. models_loaded=%s load_errors=%d",
            self.models_loaded,
            len(self.load_errors),
        )

    def shutdown(self) -> None:
        """Clean up runtime state on application shutdown."""

        LOGGER.info("Shutting down API runtime.")

    def correct(
        self,
        text: str,
        mode: str = "auto",
        num_beams: int = 4,
        return_detected_errors: bool = False,
        prompt_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Correct a single text input with guardrails and runtime fallbacks.

        Args:
            text: User input to correct.
            mode: Correction mode, one of `t5`, `rag`, or `auto`.
            num_beams: Beam width used for T5 decoding when available.
            return_detected_errors: Whether to include token-level error spans.
            prompt_version: Optional prompt version id for RAG mode.

        Returns:
            Dict[str, Any]: API-ready correction payload.
        """

        started = perf_counter()
        report = self._validate_request(text)
        sanitized_text = report.input_valid.sanitized_text
        detected_errors = self._detect_errors(sanitized_text)

        if mode == "auto" and not detected_errors:
            corrected = sanitized_text
            mode_used = "auto"
        elif mode == "rag":
            corrected = self._rag_correct(sanitized_text, prompt_version=prompt_version)
            mode_used = "rag"
        else:
            corrected = self._t5_correct(sanitized_text, num_beams=num_beams)
            mode_used = "t5" if mode == "t5" else "auto"

        report = self._finalize_report(sanitized_text, corrected)
        if not report.output_valid or not report.output_valid.passed:
            message = (
                "; ".join(report.output_valid.violations)
                if report.output_valid
                else ("Output validation failed.")
            )
            raise GuardrailViolation("output_validation", message, "error")

        return {
            "original": sanitized_text,
            "corrected": report.output_valid.sanitized_text,
            "mode_used": mode_used,
            "errors_detected": detected_errors if return_detected_errors else None,
            "guardrail_report": report,
            "processing_time_ms": round((perf_counter() - started) * 1000, 3),
            "prompt_version": prompt_version
            or self.prompt_manager.get_active_prompt().version_id,
        }

    def correct_batch(
        self,
        texts: List[str],
        mode: str = "auto",
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """Correct a batch of text inputs without aborting on per-item failures.

        Args:
            texts: Input texts to process.
            mode: Correction mode shared by the batch.
            batch_size: Batch size hint recorded in the response.

        Returns:
            List[Dict[str, Any]]: Batch item results with success or error status.
        """

        results: List[Dict[str, Any]] = []
        for text in texts:
            item_started = perf_counter()
            try:
                corrected = self.correct(
                    text, mode=mode, num_beams=self.config.model.num_beams
                )
                results.append(
                    {
                        "original": corrected["original"],
                        "corrected": corrected["corrected"],
                        "mode_used": corrected["mode_used"],
                        "status": "success",
                        "error": None,
                        "processing_time_ms": corrected["processing_time_ms"],
                        "batch_size": batch_size,
                    }
                )
            except (
                Exception
            ) as exc:  # pragma: no cover - error path exercised in API tests
                results.append(
                    {
                        "original": text,
                        "corrected": text,
                        "mode_used": mode,
                        "status": "error",
                        "error": str(exc),
                        "processing_time_ms": round(
                            (perf_counter() - item_started) * 1000, 3
                        ),
                        "batch_size": batch_size,
                    }
                )
        return results

    def detect(self, text: str) -> Dict[str, Any]:
        """Detect token-level error spans in a user input.

        Args:
            text: User input to inspect.

        Returns:
            Dict[str, Any]: Detection payload with spans and timing.
        """

        started = perf_counter()
        report = self._validate_request(text)
        spans = self._detect_errors(report.input_valid.sanitized_text)
        return {
            "has_errors": bool(spans),
            "errors": spans,
            "error_count": len(spans),
            "processing_time_ms": round((perf_counter() - started) * 1000, 3),
        }

    def add_knowledge_rules(self, rules: List[str]) -> Dict[str, int]:
        """Add grammar rules to the knowledge base and rebuild the index.

        Args:
            rules: Rule strings to add.

        Returns:
            Dict[str, int]: Count of added rules and total stored rules.
        """

        cleaned_rules = [rule.strip() for rule in rules if str(rule).strip()]
        if not cleaned_rules:
            raise ValueError("At least one non-empty grammar rule is required.")
        self._ensure_knowledge_base()
        self.rag.add_grammar_rules(cleaned_rules)
        return {"added": len(cleaned_rules), "total_rules": len(self.rag._documents)}

    def search_knowledge(self, query: str, top_k: int) -> Dict[str, Any]:
        """Search the grammar knowledge base for relevant rules.

        Args:
            query: Search text.
            top_k: Maximum number of retrieved chunks.

        Returns:
            Dict[str, Any]: Search query and retrieved chunks.
        """

        if not query.strip():
            raise ValueError("Query text cannot be empty.")
        self._ensure_knowledge_base()
        return {"query": query, "results": self.rag.retrieve(query, top_k=top_k)}

    def list_prompt_versions(self) -> Dict[str, Any]:
        """List all registered prompts and the current active version."""

        versions = self.prompt_manager.list_versions()
        active_version = self.prompt_manager.get_active_prompt().version_id
        return {"versions": versions, "active_version": active_version}

    def get_prompt_version(self, version_id: str) -> Any:
        """Return a single prompt version.

        Args:
            version_id: Semantic version identifier.

        Returns:
            Any: Prompt version dataclass.
        """

        return self.prompt_manager.get_prompt(version_id)

    def promote_prompt(self, version_id: str) -> Dict[str, str]:
        """Promote a prompt version to active status.

        Args:
            version_id: Semantic version identifier to activate.

        Returns:
            Dict[str, str]: Promotion metadata.
        """

        previous = self.prompt_manager.get_active_prompt().version_id
        self.prompt_manager.promote_prompt(version_id)
        return {"promoted": version_id, "previous": previous}

    def rollback_prompt(self) -> Dict[str, str]:
        """Rollback the active prompt to the previous version."""

        prompt = self.prompt_manager.rollback()
        return {"rolled_back_to": prompt.version_id}

    def evaluate_metrics(
        self,
        predictions: List[str],
        references: List[str],
        metrics: List[str],
    ) -> Dict[str, Any]:
        """Evaluate prediction/reference pairs using requested metrics.

        Args:
            predictions: Corrected outputs to score.
            references: Ground-truth references.
            metrics: Requested metric names.

        Returns:
            Dict[str, Any]: Metric payload and evaluated pair count.
        """

        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length.")

        results: Dict[str, Any] = {}
        for metric in metrics:
            if metric == "exact_match":
                results["exact_match"] = self.evaluator.compute_exact_match(
                    predictions,
                    references,
                )
            elif metric == "gleu":
                try:
                    results["gleu"] = self.evaluator.compute_gleu(
                        predictions,
                        [[reference] for reference in references],
                    )
                except ImportError:
                    results["gleu"] = self._fallback_gleu(predictions, references)
            elif metric == "rouge":
                try:
                    results["rouge"] = self.evaluator.compute_rouge(
                        predictions,
                        references,
                    )
                except ImportError:
                    results["rouge"] = self._fallback_rouge(predictions, references)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        return {"metrics": results, "evaluated_pairs": len(predictions)}

    def serialize(self, value: Any) -> Any:
        """Convert runtime objects into JSON-serializable data.

        Args:
            value: Runtime object to serialize.

        Returns:
            Any: Plain Python data structure.
        """

        if is_dataclass(value):
            return {key: self.serialize(item) for key, item in asdict(value).items()}
        if isinstance(value, list):
            return [self.serialize(item) for item in value]
        if isinstance(value, dict):
            return {key: self.serialize(item) for key, item in value.items()}
        return value

    def _try_load_model(self, label: str, loader: Any) -> bool:
        """Attempt to load a model, recording any failure for health reporting."""

        try:
            loader()
            return True
        except Exception as exc:  # pragma: no cover - exercised by runtime startup
            message = f"{label} load failed: {exc}"
            LOGGER.warning(message)
            self.load_errors.append(message)
            return False

    def _ensure_knowledge_base(self) -> None:
        """Load or build the default grammar-rule knowledge base."""

        metadata_path = self.config.rag.vector_store_path / "metadata.json"
        chunks_path = self.config.rag.vector_store_path / "chunks.json"
        if metadata_path.exists() and chunks_path.exists():
            self.rag.load_knowledge_base()
            return

        rules_path = self.config.data.sample_data_path / "grammar_rules.txt"
        if not rules_path.exists():
            raise FileNotFoundError(
                f"Default grammar knowledge base is missing: {rules_path}"
            )
        rules = [
            line.strip()
            for line in rules_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.rag.build_knowledge_base(rules)

    def _validate_request(self, text: str) -> FullGuardrailReport:
        """Run input guardrails and raise on blocking violations."""

        report = self.guardrails.run_all_checks(input_text=text)
        if not report.input_valid.passed:
            raise GuardrailViolation(
                "input_validation",
                "; ".join(report.input_valid.violations),
                report.input_valid.severity,
            )
        if report.toxicity.is_toxic:
            raise GuardrailViolation(
                "toxicity",
                "Input was blocked by the toxicity filter.",
                "error",
            )
        return report

    def _finalize_report(self, original: str, corrected: str) -> FullGuardrailReport:
        """Run full input and output guardrail checks."""

        return self.guardrails.run_all_checks(
            input_text=original,
            output_text=corrected,
        )

    def _detect_errors(self, text: str) -> List[ErrorSpan]:
        """Detect errors with BERT when available, otherwise use heuristics."""

        if self.bert.model is not None and self.bert.tokenizer is not None:
            return self.bert.detect_errors(text)
        return self._heuristic_detect_errors(text)

    def _t5_correct(self, text: str, num_beams: int) -> str:
        """Run T5 correction when available, otherwise use heuristic rules."""

        if self.t5.model is not None and self.t5.tokenizer is not None:
            return self.t5.correct(
                text, num_beams=num_beams, max_length=self.t5.max_length
            )
        return self._heuristic_correct(text)

    def _rag_correct(self, text: str, prompt_version: Optional[str]) -> str:
        """Run RAG prompt augmentation and correction using the best available path."""

        prompt = (
            self.prompt_manager.get_prompt(prompt_version)
            if prompt_version
            else self.prompt_manager.get_active_prompt()
        )
        return self.rag.rag_correct(
            text,
            llm_fn=lambda _: self._t5_correct(
                text,
                num_beams=self.config.model.num_beams,
            ),
            template=prompt.template,
        )

    def _heuristic_correct(self, text: str) -> str:
        """Apply lightweight grammar heuristics when models are unavailable."""

        corrected = " ".join(text.split())
        corrected = re.sub(r"\beveryday\b", "every day", corrected, flags=re.IGNORECASE)
        corrected = re.sub(r"\bteh\b", "the", corrected, flags=re.IGNORECASE)
        corrected = re.sub(r"\brecieve\b", "receive", corrected, flags=re.IGNORECASE)
        corrected = re.sub(r"\bseperate\b", "separate", corrected, flags=re.IGNORECASE)
        corrected = re.sub(
            r"\ba ([aeiouAEIOU]\w+)",
            lambda match: f"an {match.group(1)}",
            corrected,
        )
        corrected = re.sub(
            r"\ban ([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\w+)",
            lambda match: f"a {match.group(1)}",
            corrected,
        )
        corrected = self._fix_singular_pronoun_verbs(corrected)
        corrected = self._fix_plural_pronoun_verbs(corrected)
        corrected = self._fix_past_tense_markers(corrected)
        return corrected

    def _heuristic_detect_errors(self, text: str) -> List[ErrorSpan]:
        """Detect likely error spans by comparing text with heuristic correction."""

        corrected = self._heuristic_correct(text)
        if corrected == text:
            return []

        source_matches = list(re.finditer(r"\S+", text))
        source_tokens = [match.group(0) for match in source_matches]
        target_tokens = corrected.split()
        matcher = SequenceMatcher(a=source_tokens, b=target_tokens)
        spans: List[ErrorSpan] = []
        for opcode, src_start, src_end, _, _ in matcher.get_opcodes():
            if opcode == "equal":
                continue
            for token_index in range(src_start, src_end):
                match = source_matches[token_index]
                spans.append(
                    ErrorSpan(
                        start=match.start(),
                        end=match.end(),
                        token=match.group(0),
                        confidence=0.85,
                        error_type="HEURISTIC_GRAMMAR_ERROR",
                    )
                )
        return spans

    def _fix_singular_pronoun_verbs(self, text: str) -> str:
        """Add third-person singular inflection for simple present verbs."""

        def replace(match: re.Match[str]) -> str:
            pronoun = match.group(1)
            verb = match.group(2)
            lower_verb = verb.casefold()
            irregular = {"go": "goes", "do": "does", "have": "has"}
            inflected = irregular.get(lower_verb) or self._inflect_present_singular(
                verb
            )
            return f"{pronoun} {inflected}"

        pattern = r"\b(He|She|It|he|she|it)\s+([A-Za-z']+)\b"
        return re.sub(pattern, replace, text)

    def _fix_plural_pronoun_verbs(self, text: str) -> str:
        """Remove singular inflection from simple present plural pronoun verbs."""

        def replace(match: re.Match[str]) -> str:
            pronoun = match.group(1)
            verb = match.group(2)
            lowered = verb.casefold()
            irregular = {"does": "do", "has": "have", "goes": "go"}
            if lowered in irregular:
                return f"{pronoun} {irregular[lowered]}"
            if lowered.endswith("ies") and len(verb) > 3:
                return f"{pronoun} {verb[:-3]}y"
            if lowered.endswith("es") and len(verb) > 2:
                return f"{pronoun} {verb[:-2]}"
            if lowered.endswith("s") and len(verb) > 1:
                return f"{pronoun} {verb[:-1]}"
            return match.group(0)

        pattern = r"\b(They|We|You|they|we|you)\s+([A-Za-z']+)\b"
        return re.sub(pattern, replace, text)

    def _fix_past_tense_markers(self, text: str) -> str:
        """Convert a few common present-tense verbs after past-time markers."""

        replacements = {
            "go": "went",
            "come": "came",
            "eat": "ate",
            "buy": "bought",
            "see": "saw",
            "take": "took",
            "make": "made",
        }
        pattern = (
            r"\b(yesterday|last night|last week|last year)\b"
            r"([^.!?]*?)\b(I|We|They|He|She|i|we|they|he|she)\s+([A-Za-z']+)\b"
        )

        def replace(match: re.Match[str]) -> str:
            marker = match.group(1)
            middle = match.group(2)
            subject = match.group(3)
            verb = match.group(4)
            corrected = replacements.get(verb.casefold(), verb)
            return f"{marker}{middle}{subject} {corrected}"

        return re.sub(pattern, replace, text)

    def _inflect_present_singular(self, verb: str) -> str:
        """Inflect a regular verb for third-person singular present tense."""

        lower_verb = verb.casefold()
        if lower_verb in {"is", "was", "has", "does", "goes"}:
            return verb
        if lower_verb.endswith(("ch", "sh", "x", "s", "z", "o")):
            return f"{verb}es"
        if lower_verb.endswith("y") and len(verb) > 1 and lower_verb[-2] not in "aeiou":
            return f"{verb[:-1]}ies"
        return f"{verb}s"

    def _fallback_gleu(self, predictions: List[str], references: List[str]) -> float:
        """Approximate GLEU with an average sequence-similarity ratio."""

        if not predictions:
            return 0.0
        total = 0.0
        for prediction, reference in zip(predictions, references):
            total += SequenceMatcher(
                a=reference.casefold().split(),
                b=prediction.casefold().split(),
            ).ratio()
        return total / len(predictions)

    def _fallback_rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Approximate ROUGE metrics with token-overlap heuristics."""

        if not predictions:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        rouge1_total = 0.0
        rouge2_total = 0.0
        rouge_l_total = 0.0
        for prediction, reference in zip(predictions, references):
            prediction_tokens = prediction.casefold().split()
            reference_tokens = reference.casefold().split()
            overlap = len(set(prediction_tokens) & set(reference_tokens))
            rouge1_total += overlap / max(len(set(reference_tokens)), 1)

            prediction_bigrams = self._bigrams(prediction_tokens)
            reference_bigrams = self._bigrams(reference_tokens)
            bigram_overlap = len(prediction_bigrams & reference_bigrams)
            rouge2_total += bigram_overlap / max(len(reference_bigrams), 1)

            rouge_l_total += SequenceMatcher(
                a=reference_tokens,
                b=prediction_tokens,
            ).ratio()

        count = len(predictions)
        return {
            "rouge1": rouge1_total / count,
            "rouge2": rouge2_total / count,
            "rougeL": rouge_l_total / count,
        }

    def _bigrams(self, tokens: Iterable[str]) -> set[tuple[str, str]]:
        """Build token bigrams for lightweight ROUGE-2 approximation."""

        token_list = list(tokens)
        return {
            (token_list[index], token_list[index + 1])
            for index in range(len(token_list) - 1)
        }


def utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()

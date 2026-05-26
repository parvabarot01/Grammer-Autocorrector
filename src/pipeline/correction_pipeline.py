"""Unified correction pipeline for the Grammar Autocorrector system."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.models import (
    BERTGrammarDetector,
    ErrorSpan,
    RNNGrammarCorrector,
    T5GrammarCorrector,
)
from src.pipeline.guardrails import (
    FullGuardrailReport,
    GrammarGuardrails,
    GuardrailViolation,
)
from src.pipeline.prompt_versioning import PromptVersionManager
from src.pipeline.rag_pipeline import GrammarRAGPipeline, RetrievedChunk
from src.utils.config import Config
from src.utils.evaluation import Evaluator

LOGGER = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Represent the output of one correction request."""

    original: str
    corrected: str
    mode_used: str
    errors_detected: Optional[List[ErrorSpan]]
    guardrail_report: FullGuardrailReport
    processing_time_ms: float
    model_version: str
    prompt_version: str
    status: str = "success"
    error_message: Optional[str] = None


@dataclass
class BenchmarkReport:
    """Represent benchmark metrics for a batch of correction requests."""

    gleu: float
    rouge: Dict[str, float]
    exact_match: float
    avg_latency_ms: float
    p95_latency_ms: float
    failure_rate: float
    total_samples: int
    timestamp: str


class CorrectionPipeline:
    """Unified pipeline orchestrating detection, correction, and guardrails."""

    def __init__(self, config: Config) -> None:
        """Initialize the pipeline and its component services.

        Args:
            config: Project configuration object.
        """

        self.config = config
        self.t5 = T5GrammarCorrector(config.model.t5_model_name)
        self.bert = BERTGrammarDetector(config.model.bert_model_name)
        self.rnn = RNNGrammarCorrector()
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
        self._models_loaded = False
        self.load_errors: List[str] = []

    @property
    def models_loaded(self) -> bool:
        """Return whether the primary T5 and BERT models loaded successfully."""

        return self._models_loaded

    def load_all(self) -> None:
        """Load models and initialize the retrieval knowledge base."""

        self.load_errors = []
        self._load_component("rag", self._ensure_knowledge_base)
        t5_loaded = self._load_component("t5", self.t5.load_model)
        bert_loaded = self._load_component("bert", self.bert.load_model)
        self._models_loaded = t5_loaded and bert_loaded
        LOGGER.info(
            "Correction pipeline ready. models_loaded=%s load_errors=%d",
            self._models_loaded,
            len(self.load_errors),
        )

    def correct(
        self,
        text: str,
        mode: str = "auto",
        num_beams: int = 4,
        return_errors: bool = False,
        prompt_version: str | None = None,
    ) -> CorrectionResult:
        """Correct a single input text.

        Args:
            text: Input sentence or paragraph to correct.
            mode: One of `auto`, `t5`, or `rag`.
            num_beams: Beam width for T5 decoding.
            return_errors: Whether to attach token-level error spans.
            prompt_version: Optional prompt version for RAG mode.

        Returns:
            CorrectionResult: Structured correction output with metadata.
        """

        started = perf_counter()
        input_report = self.guardrails.run_all_checks(input_text=text)
        self._raise_for_input_violations(input_report)
        sanitized_text = input_report.input_valid.sanitized_text
        resolved_prompt_version = (
            prompt_version or self.prompt_manager.get_active_prompt().version_id
        )

        detected_errors = self.detect_errors(sanitized_text)
        if mode == "auto" and not detected_errors:
            corrected_text = sanitized_text
            mode_used = "auto"
        elif mode == "rag":
            corrected_text = self._rag_correct(
                sanitized_text,
                prompt_version=resolved_prompt_version,
            )
            mode_used = "rag"
        else:
            corrected_text = self._t5_correct(sanitized_text, num_beams=num_beams)
            mode_used = "t5" if mode == "t5" else "auto"

        final_report = self.guardrails.run_all_checks(
            input_text=sanitized_text,
            output_text=corrected_text,
        )
        if final_report.output_valid is None or not final_report.output_valid.passed:
            message = (
                "; ".join(final_report.output_valid.violations)
                if final_report.output_valid is not None
                else "Output validation failed."
            )
            raise GuardrailViolation("output_validation", message, "error")

        return CorrectionResult(
            original=sanitized_text,
            corrected=final_report.output_valid.sanitized_text,
            mode_used=mode_used,
            errors_detected=detected_errors if return_errors else None,
            guardrail_report=final_report,
            processing_time_ms=round((perf_counter() - started) * 1000, 3),
            model_version=self.t5.model_name,
            prompt_version=resolved_prompt_version,
        )

    def correct_batch(
        self,
        texts: List[str],
        mode: str = "auto",
        batch_size: int = 16,
    ) -> List[CorrectionResult]:
        """Correct a batch of texts without aborting on per-item failures.

        Args:
            texts: Input texts to correct.
            mode: Shared correction mode for the batch.
            batch_size: Number of items per processing chunk.

        Returns:
            List[CorrectionResult]: Per-item results, including failures.
        """

        results: List[CorrectionResult] = []
        for chunk_start in range(0, len(texts), batch_size):
            chunk = texts[chunk_start : chunk_start + batch_size]
            for text in chunk:
                item_started = perf_counter()
                try:
                    results.append(self.correct(text, mode=mode, return_errors=False))
                except Exception as exc:
                    report = self.guardrails.run_all_checks(input_text=text)
                    results.append(
                        CorrectionResult(
                            original=text,
                            corrected=text,
                            mode_used=mode,
                            errors_detected=None,
                            guardrail_report=report,
                            processing_time_ms=round(
                                (perf_counter() - item_started) * 1000,
                                3,
                            ),
                            model_version=self.t5.model_name,
                            prompt_version=self.prompt_manager.get_active_prompt().version_id,
                            status="error",
                            error_message=str(exc),
                        )
                    )
        return results

    def benchmark(self, test_data: List[Tuple[str, str]]) -> BenchmarkReport:
        """Run correction over benchmark pairs and compute aggregate metrics.

        Args:
            test_data: Sequence of `(original, reference)` pairs.

        Returns:
            BenchmarkReport: Aggregate quality and latency statistics.
        """

        if not test_data:
            raise ValueError("Benchmark data cannot be empty.")

        predictions: List[str] = []
        references: List[str] = []
        latencies: List[float] = []
        failures = 0

        for original, reference in test_data:
            try:
                result = self.correct(original, mode="auto", return_errors=False)
                predictions.append(result.corrected)
                latencies.append(result.processing_time_ms)
            except Exception:
                failures += 1
                predictions.append(original)
                latencies.append(0.0)
            references.append(reference)

        try:
            gleu = self.evaluator.compute_gleu(
                predictions,
                [[reference] for reference in references],
            )
        except ImportError:
            gleu = self._fallback_gleu(predictions, references)

        try:
            rouge = self.evaluator.compute_rouge(predictions, references)
        except ImportError:
            rouge = self._fallback_rouge(predictions, references)

        exact_match = self.evaluator.compute_exact_match(predictions, references)
        return BenchmarkReport(
            gleu=gleu,
            rouge=rouge,
            exact_match=exact_match,
            avg_latency_ms=fmean(latencies),
            p95_latency_ms=self._percentile(latencies, 95),
            failure_rate=failures / len(test_data),
            total_samples=len(test_data),
            timestamp=self.utcnow(),
        )

    def detect_errors(self, text: str) -> List[ErrorSpan]:
        """Detect token-level errors in a text input."""

        if self.bert.model is not None and self.bert.tokenizer is not None:
            return self.bert.detect_errors(text)
        return self._heuristic_detect_errors(text)

    def detect(self, text: str) -> Dict[str, Any]:
        """Return an API-friendly error-detection payload."""

        started = perf_counter()
        report = self.guardrails.run_all_checks(input_text=text)
        self._raise_for_input_violations(report)
        spans = self.detect_errors(report.input_valid.sanitized_text)
        return {
            "has_errors": bool(spans),
            "errors": spans,
            "error_count": len(spans),
            "processing_time_ms": round((perf_counter() - started) * 1000, 3),
        }

    def add_knowledge_rules(self, rules: List[str]) -> Dict[str, int]:
        """Add rules to the knowledge base and return rule counts."""

        cleaned_rules = [rule.strip() for rule in rules if str(rule).strip()]
        if not cleaned_rules:
            raise ValueError("At least one non-empty grammar rule is required.")
        self._ensure_knowledge_base()
        self.rag.add_grammar_rules(cleaned_rules)
        return {"added": len(cleaned_rules), "total_rules": len(self.rag._documents)}

    def search_knowledge(self, query: str, top_k: int) -> Dict[str, Any]:
        """Search the knowledge base for relevant grammar rules."""

        if not query.strip():
            raise ValueError("Query text cannot be empty.")
        self._ensure_knowledge_base()
        return {"query": query, "results": self.rag.retrieve(query, top_k=top_k)}

    def list_prompt_versions(self) -> Dict[str, Any]:
        """List prompt versions and the active version."""

        return {
            "versions": self.prompt_manager.list_versions(),
            "active_version": self.prompt_manager.get_active_prompt().version_id,
        }

    def get_prompt_version(self, version_id: str) -> Any:
        """Return one prompt version by id."""

        return self.prompt_manager.get_prompt(version_id)

    def promote_prompt(self, version_id: str) -> Dict[str, str]:
        """Promote a prompt version to active status."""

        previous = self.prompt_manager.get_active_prompt().version_id
        self.prompt_manager.promote_prompt(version_id)
        return {"promoted": version_id, "previous": previous}

    def rollback_prompt(self) -> Dict[str, str]:
        """Rollback to the previous active prompt."""

        prompt = self.prompt_manager.rollback()
        return {"rolled_back_to": prompt.version_id}

    def evaluate_metrics(
        self,
        predictions: List[str],
        references: List[str],
        metrics: List[str],
    ) -> Dict[str, Any]:
        """Evaluate predictions against references with selected metrics."""

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
        """Convert nested dataclasses into JSON-serializable structures."""

        if is_dataclass(value):
            return {key: self.serialize(item) for key, item in asdict(value).items()}
        if isinstance(value, list):
            return [self.serialize(item) for item in value]
        if isinstance(value, dict):
            return {key: self.serialize(item) for key, item in value.items()}
        return value

    def _load_component(self, label: str, loader: Any) -> bool:
        """Attempt to load one component and track failures."""

        started = perf_counter()
        try:
            loader()
            LOGGER.info(
                "%s loaded in %.2f ms", label, (perf_counter() - started) * 1000
            )
            return True
        except Exception as exc:
            message = f"{label} load failed: {exc}"
            LOGGER.warning(message)
            self.load_errors.append(message)
            return False

    def _ensure_knowledge_base(self) -> None:
        """Load or build the grammar-rule knowledge base."""

        metadata_path = self.config.rag.vector_store_path / "metadata.json"
        chunks_path = self.config.rag.vector_store_path / "chunks.json"
        if metadata_path.exists() and chunks_path.exists():
            self.rag.load_knowledge_base()
            return

        rules_path = Path(self.config.data.sample_data_path) / "grammar_rules.txt"
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

    def _raise_for_input_violations(self, report: FullGuardrailReport) -> None:
        """Raise structured exceptions for blocking input issues."""

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

    def _t5_correct(self, text: str, num_beams: int) -> str:
        """Run T5 correction or a heuristic fallback."""

        if self.t5.model is not None and self.t5.tokenizer is not None:
            return self.t5.correct(
                text, num_beams=num_beams, max_length=self.t5.max_length
            )
        return self._heuristic_correct(text)

    def _rag_correct(self, text: str, prompt_version: str) -> str:
        """Run retrieval-augmented correction using the chosen prompt version."""

        prompt = self.prompt_manager.get_prompt(prompt_version)
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
        """Detect likely error spans by diffing heuristic corrections."""

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
            irregular = {"go": "goes", "do": "does", "have": "has"}
            inflected = irregular.get(
                verb.casefold()
            ) or self._inflect_present_singular(verb)
            return f"{pronoun} {inflected}"

        pattern = r"\b(He|She|It|he|she|it)\s+([A-Za-z']+)\b"
        return re.sub(pattern, replace, text)

    def _fix_plural_pronoun_verbs(self, text: str) -> str:
        """Remove singular inflection from plural-subject present verbs."""

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
        """Convert a few common verbs after explicit past-time markers."""

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
        """Approximate GLEU using token-level sequence similarity."""

        if not predictions:
            return 0.0
        scores = [
            SequenceMatcher(
                a=reference.casefold().split(),
                b=prediction.casefold().split(),
            ).ratio()
            for prediction, reference in zip(predictions, references)
        ]
        return fmean(scores)

    def _fallback_rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Approximate ROUGE metrics with token overlap and sequence similarity."""

        if not predictions:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        rouge1_scores: List[float] = []
        rouge2_scores: List[float] = []
        rouge_l_scores: List[float] = []
        for prediction, reference in zip(predictions, references):
            prediction_tokens = prediction.casefold().split()
            reference_tokens = reference.casefold().split()
            overlap = len(set(prediction_tokens) & set(reference_tokens))
            rouge1_scores.append(overlap / max(len(set(reference_tokens)), 1))

            prediction_bigrams = self._bigrams(prediction_tokens)
            reference_bigrams = self._bigrams(reference_tokens)
            bigram_overlap = len(prediction_bigrams & reference_bigrams)
            rouge2_scores.append(bigram_overlap / max(len(reference_bigrams), 1))

            rouge_l_scores.append(
                SequenceMatcher(a=reference_tokens, b=prediction_tokens).ratio()
            )

        return {
            "rouge1": fmean(rouge1_scores),
            "rouge2": fmean(rouge2_scores),
            "rougeL": fmean(rouge_l_scores),
        }

    def _bigrams(self, tokens: Iterable[str]) -> set[Tuple[str, str]]:
        """Build token bigrams for the ROUGE-2 fallback."""

        token_list = list(tokens)
        return {
            (token_list[index], token_list[index + 1])
            for index in range(len(token_list) - 1)
        }

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Compute a percentile from a list of latency values."""

        if not values:
            return 0.0
        ordered = sorted(values)
        rank = max(math.ceil((percentile / 100.0) * len(ordered)) - 1, 0)
        return ordered[min(rank, len(ordered) - 1)]

    def utcnow(self) -> str:
        """Return the current UTC timestamp."""

        return datetime.now(timezone.utc).isoformat()

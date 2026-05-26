"""Pydantic request and response models for the FastAPI grammar API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CorrectionMode = Literal["t5", "rag", "auto"]
MetricName = Literal["gleu", "rouge", "exact_match"]


class ErrorSpanModel(BaseModel):
    """Serializable error-span schema."""

    start: int = Field(..., description="Start character offset of the error span.")
    end: int = Field(..., description="End character offset of the error span.")
    token: str = Field(..., description="Original token or span text.")
    confidence: float = Field(..., description="Model confidence for the error label.")
    error_type: str = Field("UNKNOWN", description="Detected error category label.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start": 4,
                "end": 6,
                "token": "go",
                "confidence": 0.92,
                "error_type": "GRAMMAR_ERROR",
            }
        }
    )


class GuardrailResultModel(BaseModel):
    """Serializable input or output validation schema."""

    passed: bool = Field(..., description="Whether the validation stage passed.")
    violations: List[str] = Field(
        default_factory=list,
        description="Validation messages or policy violations.",
    )
    sanitized_text: str = Field(..., description="Sanitized text passed downstream.")
    severity: Literal["none", "warning", "error"] = Field(
        ...,
        description="Most severe validation level observed in this stage.",
    )


class ToxicityResultModel(BaseModel):
    """Serializable toxicity-scan schema."""

    score: float = Field(
        ..., description="Keyword-based toxicity score between 0 and 1."
    )
    is_toxic: bool = Field(
        ..., description="Whether the text exceeded the toxicity threshold."
    )
    detected_terms: List[str] = Field(
        default_factory=list,
        description="Matched toxicity keywords.",
    )


class BiasResultModel(BaseModel):
    """Serializable bias-scan schema."""

    has_bias: bool = Field(
        ..., description="Whether simple bias patterns were detected."
    )
    bias_types: List[str] = Field(
        default_factory=list,
        description="Matched bias categories.",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggested wording improvements.",
    )


class FullGuardrailReportModel(BaseModel):
    """Serializable aggregate guardrail report."""

    input_valid: GuardrailResultModel = Field(
        ...,
        description="Input validation result.",
    )
    toxicity: ToxicityResultModel = Field(
        ...,
        description="Keyword-based toxicity analysis.",
    )
    bias: BiasResultModel = Field(..., description="Bias-scan analysis.")
    output_valid: Optional[GuardrailResultModel] = Field(
        None,
        description="Output validation result when a model response is available.",
    )
    overall_passed: bool = Field(
        ...,
        description="Overall pass/fail state across enabled checks.",
    )
    timestamp: str = Field(..., description="UTC timestamp for the report.")


class HealthResponse(BaseModel):
    """Health endpoint response model."""

    status: str = Field(..., description="Service status string.")
    models_loaded: bool = Field(
        ...,
        description="Whether T5 and BERT model weights were loaded successfully.",
    )
    version: str = Field(..., description="API semantic version.")
    timestamp: str = Field(..., description="UTC timestamp at response time.")


class InfoResponse(BaseModel):
    """Service metadata response model."""

    models: List[str] = Field(
        ..., description="Available model and pipeline components."
    )
    prompt_version: str = Field(..., description="Currently active prompt version.")
    capabilities: List[str] = Field(..., description="Supported API capabilities.")


class CorrectionRequest(BaseModel):
    """Single-text correction request model."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Input text to correct.",
    )
    mode: CorrectionMode = Field(
        "auto",
        description="Correction mode selection.",
    )
    num_beams: int = Field(
        4,
        ge=1,
        le=8,
        description="Beam width for T5 decoding.",
    )
    return_detected_errors: bool = Field(
        False,
        description="Whether to include token-level error spans in the response.",
    )
    prompt_version: Optional[str] = Field(
        None,
        description="Optional semantic prompt version for RAG mode.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "She go to school everyday.",
                "mode": "auto",
                "num_beams": 4,
                "return_detected_errors": True,
                "prompt_version": "v1.1.0",
            }
        }
    )


class CorrectionResponse(BaseModel):
    """Single-text correction response model."""

    original: str = Field(..., description="Sanitized input text.")
    corrected: str = Field(..., description="Corrected output text.")
    mode_used: str = Field(..., description="Correction mode actually used.")
    errors_detected: Optional[List[ErrorSpanModel]] = Field(
        None,
        description="Detected token-level errors when requested.",
    )
    guardrail_report: FullGuardrailReportModel = Field(
        ...,
        description="Input and output guardrail report.",
    )
    processing_time_ms: float = Field(
        ...,
        description="End-to-end processing time in milliseconds.",
    )
    model_version: str = Field(..., description="Model or checkpoint identifier used.")
    prompt_version: str = Field(..., description="Resolved prompt version id.")
    request_id: str = Field(..., description="Request correlation identifier.")


class BatchCorrectionRequest(BaseModel):
    """Batch correction request model."""

    texts: List[str] = Field(
        ...,
        description="Input texts to correct. Maximum of 50 items.",
    )
    mode: CorrectionMode = Field(
        "auto",
        description="Correction mode selection shared by the batch.",
    )
    batch_size: int = Field(
        16,
        ge=1,
        le=50,
        description="Requested processing batch size.",
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, value: List[str]) -> List[str]:
        """Validate batch size and per-item length."""

        if not value:
            raise ValueError("At least one input text is required.")
        if len(value) > 50:
            raise ValueError("A maximum of 50 input texts is allowed.")
        for item in value:
            if len(item) > 1000:
                raise ValueError("Each input text must be 1000 characters or fewer.")
        return value

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "She go to school everyday.",
                    "He have a apple.",
                ],
                "mode": "auto",
                "batch_size": 8,
            }
        }
    )


class BatchCorrectionItem(BaseModel):
    """One batch-correction result item."""

    original: str = Field(..., description="Original input text.")
    corrected: str = Field(..., description="Corrected output text or fallback copy.")
    mode_used: str = Field(..., description="Mode used for this item.")
    errors_detected: Optional[List[ErrorSpanModel]] = Field(
        None,
        description="Detected token-level error spans when available.",
    )
    guardrail_report: FullGuardrailReportModel = Field(
        ...,
        description="Guardrail report for this batch item.",
    )
    status: Literal["success", "error"] = Field(
        ...,
        description="Item-level processing status.",
    )
    error_message: Optional[str] = Field(
        None,
        description="Item-level error message when processing failed.",
    )
    processing_time_ms: float = Field(
        ...,
        description="Item-level processing time in milliseconds.",
    )
    model_version: str = Field(..., description="Model or checkpoint identifier used.")
    prompt_version: str = Field(..., description="Resolved prompt version id.")


class BatchCorrectionResponse(BaseModel):
    """Batch correction response model."""

    results: List[BatchCorrectionItem] = Field(
        ...,
        description="Per-item correction results.",
    )
    total_processed: int = Field(..., description="Number of processed items.")
    processing_time_ms: float = Field(
        ...,
        description="End-to-end batch processing time in milliseconds.",
    )


class DetectionRequest(BaseModel):
    """Error-detection request model."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Input text to analyze.",
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"text": "She go to school everyday."}}
    )


class DetectionResponse(BaseModel):
    """Error-detection response model."""

    has_errors: bool = Field(..., description="Whether any errors were detected.")
    errors: List[ErrorSpanModel] = Field(
        default_factory=list,
        description="Detected token-level error spans.",
    )
    error_count: int = Field(..., description="Number of detected error spans.")
    processing_time_ms: float = Field(
        ...,
        description="End-to-end processing time in milliseconds.",
    )


class KnowledgeAddRequest(BaseModel):
    """Knowledge-base rule ingestion request model."""

    rules: List[str] = Field(
        ..., description="Grammar rules to add to the knowledge base."
    )

    @field_validator("rules")
    @classmethod
    def validate_rules(cls, value: List[str]) -> List[str]:
        """Require at least one non-empty rule."""

        if not value or not any(item.strip() for item in value):
            raise ValueError("At least one non-empty grammar rule is required.")
        return value

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rules": [
                    "Use 'an' before a vowel sound.",
                    "A singular subject normally takes a singular verb.",
                ]
            }
        }
    )


class KnowledgeAddResponse(BaseModel):
    """Knowledge-base ingestion response model."""

    added: int = Field(..., description="Number of added rules.")
    total_rules: int = Field(..., description="Total number of indexed rules.")


class RetrievedChunkModel(BaseModel):
    """Serializable retrieved-chunk schema."""

    text: str = Field(..., description="Retrieved rule or chunk text.")
    score: float = Field(..., description="Distance or relevance score.")
    source: str = Field(..., description="Source identifier for the chunk.")
    chunk_id: int = Field(..., description="Chunk identifier within the vector store.")


class KnowledgeSearchResponse(BaseModel):
    """Knowledge-base search response model."""

    results: List[RetrievedChunkModel] = Field(
        default_factory=list,
        description="Retrieved knowledge chunks.",
    )
    query: str = Field(..., description="Original search query.")


class PromptVersionResponse(BaseModel):
    """Prompt-version response model."""

    version_id: str = Field(..., description="Semantic prompt version identifier.")
    template: str = Field(..., description="Prompt template text.")
    description: str = Field(..., description="Human-readable prompt summary.")
    created_at: str = Field(..., description="UTC creation timestamp.")
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stored evaluation metrics for the prompt version.",
    )
    is_active: bool = Field(
        ..., description="Whether the version is active in production."
    )


class PromptListResponse(BaseModel):
    """Prompt registry listing response model."""

    versions: List[PromptVersionResponse] = Field(
        ...,
        description="Registered prompt versions.",
    )
    active_version: str = Field(..., description="Currently active prompt version.")


class PromptPromoteResponse(BaseModel):
    """Prompt promotion response model."""

    promoted: str = Field(..., description="Newly activated prompt version.")
    previous: str = Field(..., description="Previously active prompt version.")


class PromptRollbackResponse(BaseModel):
    """Prompt rollback response model."""

    rolled_back_to: str = Field(..., description="Prompt version restored by rollback.")


class EvaluateRequest(BaseModel):
    """Evaluation request model."""

    predictions: List[str] = Field(..., description="Model predictions to evaluate.")
    references: List[str] = Field(..., description="Ground-truth references.")
    metrics: List[MetricName] = Field(
        ...,
        description="Requested metric names.",
    )

    @model_validator(mode="after")
    def validate_parallel_fields(self) -> "EvaluateRequest":
        """Ensure evaluation inputs are well-formed."""

        if len(self.predictions) != len(self.references):
            raise ValueError("predictions and references must have the same length.")
        if not self.metrics:
            raise ValueError("At least one metric must be requested.")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": ["She goes to school every day."],
                "references": ["She goes to school every day."],
                "metrics": ["gleu", "rouge", "exact_match"],
            }
        }
    )


class EvaluateResponse(BaseModel):
    """Evaluation response model."""

    metrics: Dict[str, Any] = Field(..., description="Requested metric results.")
    evaluated_pairs: int = Field(..., description="Number of evaluated examples.")


class BenchmarkPairRequest(BaseModel):
    """One benchmark sample pair."""

    original: str = Field(..., description="Erroneous source text to correct.")
    reference: str = Field(..., description="Ground-truth corrected text.")


class BenchmarkRequest(BaseModel):
    """Benchmark endpoint request model."""

    test_pairs: List[BenchmarkPairRequest] = Field(
        ...,
        description="Benchmark sample pairs of source and reference text.",
    )
    max_samples: int = Field(
        100,
        ge=1,
        le=1000,
        description="Maximum number of benchmark pairs to evaluate.",
    )

    @field_validator("test_pairs")
    @classmethod
    def validate_test_pairs(
        cls,
        value: List[BenchmarkPairRequest],
    ) -> List[BenchmarkPairRequest]:
        """Require at least one benchmark pair."""

        if not value:
            raise ValueError("At least one benchmark pair is required.")
        return value


class BenchmarkReportResponse(BaseModel):
    """Benchmark report response model."""

    gleu: float = Field(..., description="Corpus-level GLEU score.")
    rouge: Dict[str, float] = Field(..., description="ROUGE metrics dictionary.")
    exact_match: float = Field(
        ..., description="Case-insensitive exact match accuracy."
    )
    avg_latency_ms: float = Field(..., description="Average correction latency.")
    p95_latency_ms: float = Field(
        ..., description="95th percentile correction latency."
    )
    failure_rate: float = Field(..., description="Fraction of failed corrections.")
    total_samples: int = Field(..., description="Total number of benchmarked samples.")
    timestamp: str = Field(..., description="UTC timestamp for the benchmark run.")

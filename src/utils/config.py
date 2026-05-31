"""Configuration management for the Grammar Autocorrector System."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]


def _get_env_str(name: str, default: str) -> str:
    """Read a string environment variable."""

    return os.getenv(name, default)


def _get_env_int(name: str, default: int) -> int:
    """Read an integer environment variable."""

    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _get_env_float(name: str, default: float) -> float:
    """Read a float environment variable."""

    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float.") from exc


def _get_env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""

    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean.")


def _get_env_path(name: str, default: Path) -> Path:
    """Read a path environment variable."""

    raw_value = os.getenv(name)
    if not raw_value:
        return default
    return Path(raw_value).expanduser()


@dataclass(frozen=True)
class ModelConfig:
    """Model-level configuration settings."""

    t5_model_name: str = "t5-base"
    bert_model_name: str = "bert-base-uncased"
    max_length: int = 128
    num_beams: int = 4
    learning_rate: float = 3e-4
    batch_size: int = 16
    epochs: int = 5


@dataclass(frozen=True)
class DataConfig:
    """Dataset configuration settings."""

    conll2014_name: str = "conll2014"
    bea2019_name: str = "jfleg"
    jfleg_name: str = "jfleg"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    raw_data_path: Path = BASE_DIR / "data" / "raw"
    processed_data_path: Path = BASE_DIR / "data" / "processed"
    sample_data_path: Path = BASE_DIR / "data" / "sample"


@dataclass(frozen=True)
class RAGConfig:
    """Retrieval-augmented generation configuration settings."""

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_path: Path = BASE_DIR / "data" / "vector_store"
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass(frozen=True)
class APIConfig:
    """API serving configuration settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 100
    enable_public_api: bool = True
    show_model_details: bool = False
    frontend_origin: str = "http://localhost:3000"


@dataclass(frozen=True)
class GuardrailsConfig:
    """Responsible AI guardrail configuration settings."""

    toxicity_threshold: float = 0.8
    max_input_length: int = 1000


@dataclass(frozen=True)
class Config:
    """Top-level application configuration."""

    model: ModelConfig
    data: DataConfig
    rag: RAGConfig
    api: APIConfig
    guardrails: GuardrailsConfig


def _validate_split_ratios(data_config: DataConfig) -> None:
    """Validate that dataset split ratios sum to 1.0."""

    total = data_config.train_split + data_config.val_split + data_config.test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "Train, validation, and test split ratios must sum to 1.0. "
            f"Received {total:.4f}."
        )


def _ensure_directories(data_config: DataConfig, rag_config: RAGConfig) -> None:
    """Create required directories if they do not already exist."""

    for path in (
        data_config.raw_data_path,
        data_config.processed_data_path,
        data_config.sample_data_path,
        rag_config.vector_store_path,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_config(dotenv_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from environment variables and sensible defaults.

    Args:
        dotenv_path: Optional path to a dotenv file. When omitted, the function
            attempts to load `.env` from the project root.

    Returns:
        Config: The fully-populated application configuration.
    """

    env_file = Path(dotenv_path) if dotenv_path else BASE_DIR / ".env"
    load_dotenv(dotenv_path=env_file, override=False)

    model_config = ModelConfig(
        t5_model_name=_get_env_str("T5_MODEL_NAME", "t5-base"),
        bert_model_name=_get_env_str("BERT_MODEL_NAME", "bert-base-uncased"),
        max_length=_get_env_int("MODEL_MAX_LENGTH", 128),
        num_beams=_get_env_int("MODEL_NUM_BEAMS", 4),
        learning_rate=_get_env_float("MODEL_LEARNING_RATE", 3e-4),
        batch_size=_get_env_int("MODEL_BATCH_SIZE", 16),
        epochs=_get_env_int("MODEL_EPOCHS", 5),
    )

    data_config = DataConfig(
        conll2014_name=_get_env_str("CONLL2014_DATASET_NAME", "conll2014"),
        bea2019_name=_get_env_str("BEA2019_DATASET_NAME", "jfleg"),
        jfleg_name=_get_env_str("JFLEG_DATASET_NAME", "jfleg"),
        train_split=_get_env_float("DATA_TRAIN_SPLIT", 0.8),
        val_split=_get_env_float("DATA_VAL_SPLIT", 0.1),
        test_split=_get_env_float("DATA_TEST_SPLIT", 0.1),
        raw_data_path=_get_env_path("RAW_DATA_PATH", BASE_DIR / "data" / "raw"),
        processed_data_path=_get_env_path(
            "PROCESSED_DATA_PATH", BASE_DIR / "data" / "processed"
        ),
        sample_data_path=_get_env_path(
            "SAMPLE_DATA_PATH", BASE_DIR / "data" / "sample"
        ),
    )

    rag_config = RAGConfig(
        embedding_model_name=_get_env_str(
            "RAG_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        vector_store_path=_get_env_path(
            "RAG_VECTOR_STORE_PATH", BASE_DIR / "data" / "vector_store"
        ),
        top_k=_get_env_int("RAG_TOP_K", 5),
        chunk_size=_get_env_int("RAG_CHUNK_SIZE", 512),
        chunk_overlap=_get_env_int("RAG_CHUNK_OVERLAP", 50),
    )

    api_config = APIConfig(
        host=_get_env_str("API_HOST", "0.0.0.0"),
        port=_get_env_int("API_PORT", 8000),
        max_concurrent_requests=_get_env_int("API_MAX_CONCURRENT_REQUESTS", 100),
        enable_public_api=_get_env_bool("ENABLE_PUBLIC_API", True),
        show_model_details=_get_env_bool("SHOW_MODEL_DETAILS", False),
        frontend_origin=_get_env_str("FRONTEND_ORIGIN", "http://localhost:3000"),
    )

    guardrails_config = GuardrailsConfig(
        toxicity_threshold=_get_env_float("GUARDRAILS_TOXICITY_THRESHOLD", 0.8),
        max_input_length=_get_env_int("GUARDRAILS_MAX_INPUT_LENGTH", 1000),
    )

    _validate_split_ratios(data_config)
    _ensure_directories(data_config, rag_config)

    return Config(
        model=model_config,
        data=data_config,
        rag=rag_config,
        api=api_config,
        guardrails=guardrails_config,
    )

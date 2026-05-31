"""Unit tests for configuration management."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import (
    _get_env_bool,
    _get_env_float,
    _get_env_int,
    load_config,
)


def test_get_env_int_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_BATCH_SIZE", "sixteen")

    with pytest.raises(ValueError, match="MODEL_BATCH_SIZE"):
        _get_env_int("MODEL_BATCH_SIZE", 16)


def test_get_env_float_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_LEARNING_RATE", "fast")

    with pytest.raises(ValueError, match="MODEL_LEARNING_RATE"):
        _get_env_float("MODEL_LEARNING_RATE", 0.001)


def test_get_env_bool_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_PUBLIC_API", "sometimes")

    with pytest.raises(ValueError, match="ENABLE_PUBLIC_API"):
        _get_env_bool("ENABLE_PUBLIC_API", True)


def test_load_config_reads_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw-data"
    processed_dir = tmp_path / "processed-data"
    sample_dir = tmp_path / "sample-data"
    vector_dir = tmp_path / "vector-store"

    monkeypatch.setenv("MODEL_MAX_LENGTH", "256")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("ENABLE_PUBLIC_API", "false")
    monkeypatch.setenv("SHOW_MODEL_DETAILS", "true")
    monkeypatch.setenv("FRONTEND_ORIGIN", "http://localhost:3100")
    monkeypatch.setenv("RAW_DATA_PATH", str(raw_dir))
    monkeypatch.setenv("PROCESSED_DATA_PATH", str(processed_dir))
    monkeypatch.setenv("SAMPLE_DATA_PATH", str(sample_dir))
    monkeypatch.setenv("RAG_VECTOR_STORE_PATH", str(vector_dir))

    config = load_config(dotenv_path=tmp_path / "missing.env")

    assert config.model.max_length == 256
    assert config.api.port == 9000
    assert config.api.enable_public_api is False
    assert config.api.show_model_details is True
    assert config.api.frontend_origin == "http://localhost:3100"
    assert config.data.raw_data_path == raw_dir
    assert config.rag.vector_store_path == vector_dir
    assert raw_dir.exists()
    assert processed_dir.exists()
    assert sample_dir.exists()
    assert vector_dir.exists()


def test_load_config_uses_custom_dotenv_file(tmp_path: Path) -> None:
    dotenv_path = tmp_path / "custom.env"
    dotenv_path.write_text(
        "T5_MODEL_NAME=t5-small\nRAG_TOP_K=7\n",
        encoding="utf-8",
    )

    config = load_config(dotenv_path=dotenv_path)

    assert config.model.t5_model_name == "t5-small"
    assert config.rag.top_k == 7


def test_load_config_invalid_split_ratios_raise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DATA_TRAIN_SPLIT", "0.6")
    monkeypatch.setenv("DATA_VAL_SPLIT", "0.3")
    monkeypatch.setenv("DATA_TEST_SPLIT", "0.3")

    with pytest.raises(ValueError, match="must sum to 1.0"):
        load_config(dotenv_path=tmp_path / "missing.env")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

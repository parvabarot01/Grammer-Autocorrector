"""Prompt template version management for grammar correction workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PromptVersion:
    """Represent a versioned prompt template."""

    version_id: str
    template: str
    description: str
    created_at: str
    metrics: dict
    is_active: bool = False


class PromptVersionManager:
    """Manage prompt templates with semver registration and rollback."""

    def __init__(self, registry_path: str = "data/prompt_registry.json") -> None:
        """Load an existing registry or initialize a default one."""

        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry: Dict[str, Any] = {
            "versions": [],
            "events": [],
            "active_history": [],
        }
        if self.registry_path.exists():
            self._load_registry()
        else:
            self._initialize_default_registry()

    def register_prompt(
        self, template: str, description: str, version_id: str = None
    ) -> PromptVersion:
        """Register a new prompt version and persist it.

        Args:
            template: Prompt template containing placeholders.
            description: Human-readable description of the version.
            version_id: Optional semantic version identifier.

        Returns:
            PromptVersion: The registered prompt version object.
        """

        new_version_id = version_id or self._next_version_id()
        if any(
            version.version_id == new_version_id for version in self.list_versions()
        ):
            raise ValueError(f"Prompt version '{new_version_id}' already exists.")

        prompt_version = PromptVersion(
            version_id=new_version_id,
            template=template,
            description=description,
            created_at=self._now(),
            metrics={},
            is_active=False,
        )
        self.registry["versions"].append(asdict(prompt_version))
        self.registry["events"].append(
            {
                "timestamp": self._now(),
                "event": "register",
                "version_id": new_version_id,
            }
        )
        self._save_registry()
        return prompt_version

    def get_prompt(self, version_id: str) -> PromptVersion:
        """Return a specific prompt version by id."""

        for version in self.list_versions():
            if version.version_id == version_id:
                return version
        raise KeyError(f"Unknown prompt version: {version_id}")

    def get_active_prompt(self) -> PromptVersion:
        """Return the currently active production prompt."""

        for version in self.list_versions():
            if version.is_active:
                return version
        raise RuntimeError("No active prompt is configured.")

    def promote_prompt(self, version_id: str) -> None:
        """Promote a prompt version to active status."""

        previous_active = (
            self.get_active_prompt().version_id if self.registry["versions"] else None
        )
        target_found = False
        for version in self.registry["versions"]:
            if version["version_id"] == version_id:
                version["is_active"] = True
                target_found = True
            else:
                version["is_active"] = False
        if not target_found:
            raise KeyError(f"Unknown prompt version: {version_id}")

        history = self.registry.setdefault("active_history", [])
        if not history or history[-1] != version_id:
            history.append(version_id)
        self.registry["events"].append(
            {
                "timestamp": self._now(),
                "event": "promote",
                "version_id": version_id,
                "previous_active": previous_active,
            }
        )
        self._save_registry()

    def rollback(self) -> PromptVersion:
        """Rollback the active prompt to the previous active version."""

        history = self.registry.setdefault("active_history", [])
        if len(history) < 2:
            return self.get_active_prompt()

        current_version = history.pop()
        previous_version = history[-1]
        for version in self.registry["versions"]:
            version["is_active"] = version["version_id"] == previous_version
        self.registry["events"].append(
            {
                "timestamp": self._now(),
                "event": "rollback",
                "from_version": current_version,
                "to_version": previous_version,
            }
        )
        self._save_registry()
        return self.get_prompt(previous_version)

    def list_versions(self) -> List[PromptVersion]:
        """Return all prompt versions sorted by creation time descending."""

        versions = [PromptVersion(**version) for version in self.registry["versions"]]
        return sorted(versions, key=lambda item: item.created_at, reverse=True)

    def update_metrics(self, version_id: str, metrics: dict) -> None:
        """Update stored evaluation metrics for a prompt version."""

        updated = False
        for version in self.registry["versions"]:
            if version["version_id"] == version_id:
                version["metrics"] = dict(metrics)
                updated = True
                break
        if not updated:
            raise KeyError(f"Unknown prompt version: {version_id}")
        self.registry["events"].append(
            {
                "timestamp": self._now(),
                "event": "metrics_update",
                "version_id": version_id,
            }
        )
        self._save_registry()

    def export_registry(self, output_path: str) -> None:
        """Export the full prompt registry as formatted JSON."""

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.registry, indent=2), encoding="utf-8")

    def compare_versions(self, v1: str, v2: str) -> dict:
        """Return a side-by-side comparison of two prompt versions."""

        left = self.get_prompt(v1)
        right = self.get_prompt(v2)
        return {
            "left": asdict(left),
            "right": asdict(right),
            "same_template": left.template == right.template,
            "metric_keys": sorted(set(left.metrics) | set(right.metrics)),
        }

    def _load_registry(self) -> None:
        """Load the registry JSON from disk."""

        self.registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.registry.setdefault("versions", [])
        self.registry.setdefault("events", [])
        self.registry.setdefault("active_history", [])

    def _save_registry(self) -> None:
        """Persist the registry to disk."""

        self.registry_path.write_text(
            json.dumps(self.registry, indent=2),
            encoding="utf-8",
        )

    def _initialize_default_registry(self) -> None:
        """Create the default prompt registry with three initial versions."""

        self.registry = {
            "versions": [
                asdict(
                    PromptVersion(
                        version_id="v1.0.0",
                        template=(
                            "Correct the following English sentence.\n"
                            "Sentence: {input}\n"
                            "Return only the corrected sentence."
                        ),
                        description=(
                            "Simple correction prompt with no external context."
                        ),
                        created_at=self._now(),
                        metrics={"gleu": 0.61, "exact_match": 0.52},
                        is_active=False,
                    )
                ),
                asdict(
                    PromptVersion(
                        version_id="v1.1.0",
                        template=(
                            "Use the grammar guidance below to correct the sentence.\n"
                            "Context:\n{context}\n\nSentence: {input}\n"
                            "Return only the corrected sentence."
                        ),
                        description=(
                            "Context-augmented prompt for RAG-assisted correction."
                        ),
                        created_at=self._now(),
                        metrics={"gleu": 0.68, "exact_match": 0.58},
                        is_active=True,
                    )
                ),
                asdict(
                    PromptVersion(
                        version_id="v2.0.0",
                        template=(
                            "Identify the error type, reason briefly, and then provide "
                            "the corrected sentence.\nContext:\n{context}\n\n"
                            "Sentence: {input}"
                        ),
                        description=(
                            "Chain-of-thought style prompt with error type "
                            "identification."
                        ),
                        created_at=self._now(),
                        metrics={"gleu": 0.70, "exact_match": 0.60},
                        is_active=False,
                    )
                ),
            ],
            "events": [
                {
                    "timestamp": self._now(),
                    "event": "initialize_default_registry",
                }
            ],
            "active_history": ["v1.1.0"],
        }
        self._save_registry()

    def _next_version_id(self) -> str:
        """Generate the next semantic version identifier."""

        versions = [
            self._parse_version(version.version_id) for version in self.list_versions()
        ]
        if not versions:
            return "v1.0.0"
        major, minor, patch = sorted(versions)[-1]
        return f"v{major}.{minor}.{patch + 1}"

    def _parse_version(self, version_id: str) -> tuple[int, int, int]:
        """Parse a semantic version string without the leading `v`."""

        normalized = version_id.lstrip("v")
        major, minor, patch = normalized.split(".")
        return int(major), int(minor), int(patch)

    def _now(self) -> str:
        """Return the current UTC timestamp in ISO-8601 format."""

        return datetime.now(timezone.utc).isoformat()

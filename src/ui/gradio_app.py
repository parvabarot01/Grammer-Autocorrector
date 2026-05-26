"""Gradio web interface for the Grammar Autocorrector service."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import requests

from src.utils.config import load_config

BASE_DIR = Path(__file__).resolve().parents[2]
SAMPLE_SENTENCES_PATH = BASE_DIR / "data" / "sample" / "sample_sentences.txt"


def _import_gradio() -> Any:
    """Import gradio lazily with a helpful error message."""

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError(
            "gradio is required for the web interface. Install it with "
            "`pip install gradio`."
        ) from exc
    return gr


class GrammarAPIClient:
    """Small HTTP client used by the Gradio frontend."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def get(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        """Send a GET request and return JSON."""

        response = self.session.get(f"{self.base_url}{path}", timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a POST request and return JSON."""

        response = self.session.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()


def _load_examples() -> List[str]:
    """Load sample sentences from the project data directory."""

    if not SAMPLE_SENTENCES_PATH.exists():
        return ["She go to school everyday."]
    examples: List[str] = []
    for line in SAMPLE_SENTENCES_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        examples.append(stripped.split("# correct:")[0].strip())
    return examples or ["She go to school everyday."]


def _mode_to_api_value(mode_label: str) -> str:
    """Convert a UI mode label to the API enum value."""

    return {
        "Auto": "auto",
        "T5 Only": "t5",
        "RAG-Enhanced": "rag",
    }.get(mode_label, "auto")


def _highlight_text(
    text: str, errors: Sequence[Dict[str, Any]]
) -> List[Tuple[str, str | None]]:
    """Convert error spans into Gradio HighlightedText segments."""

    if not errors:
        return [(text, None)]

    segments: List[Tuple[str, str | None]] = []
    cursor = 0
    for error in sorted(errors, key=lambda item: int(item["start"])):
        start = max(int(error["start"]), 0)
        end = max(int(error["end"]), start)
        if start > cursor:
            segments.append((text[cursor:start], None))
        segments.append((text[start:end], "Error"))
        cursor = end
    if cursor < len(text):
        segments.append((text[cursor:], None))
    return segments


def _safe_json_error(exc: Exception) -> Dict[str, Any]:
    """Normalize request errors for JSON UI display."""

    response = getattr(exc, "response", None)
    if response is None:
        return {"error": str(exc)}
    try:
        return response.json()
    except Exception:
        return {"error": response.text or str(exc)}


def _build_metric_plot(metrics: Dict[str, Any]) -> Any:
    """Build a simple bar chart for evaluation metrics when matplotlib exists."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional runtime dependency
        return None

    flattened: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (int, float)):
                    flattened[f"{key}.{nested_key}"] = float(nested_value)
        elif isinstance(value, (int, float)):
            flattened[key] = float(value)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(list(flattened.keys()), list(flattened.values()), color="#1f77b4")
    axis.set_ylim(0, max(flattened.values(), default=1.0) * 1.1)
    axis.set_title("Evaluation Metrics")
    axis.set_ylabel("Score")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    return figure


def build_app() -> Any:
    """Build the Gradio interface and wire handlers to the FastAPI service."""

    gr = _import_gradio()
    config = load_config()
    api_url = os.getenv("API_URL", f"http://{config.api.host}:{config.api.port}")
    client = GrammarAPIClient(api_url)
    examples = _load_examples()

    def correct_text_ui(
        text: str,
        mode_label: str,
        num_beams: int,
        show_errors: bool,
    ) -> tuple[str, list[tuple[str, str | None]], dict[str, Any], str]:
        """Submit a single-text correction request."""

        try:
            response = client.post(
                "/correct",
                {
                    "text": text,
                    "mode": _mode_to_api_value(mode_label),
                    "num_beams": int(num_beams),
                    "return_detected_errors": bool(show_errors),
                },
            )
            highlights = _highlight_text(
                response["original"],
                response.get("errors_detected") or [],
            )
            return (
                response["corrected"],
                highlights,
                response["guardrail_report"],
                f"{response['processing_time_ms']:.2f} ms",
            )
        except Exception as exc:  # pragma: no cover - UI error path
            error_payload = _safe_json_error(exc)
            return text, [(text, "Error")], error_payload, "Request failed"

    def correct_batch_ui(
        dataframe: List[List[Any]], mode_label: str
    ) -> tuple[list[list[Any]], str | None]:
        """Submit a batch correction request and export CSV results."""

        texts = [
            str(row[0]).strip() for row in dataframe if row and str(row[0]).strip()
        ]
        if not texts:
            return [], None
        response = client.post(
            "/correct/batch",
            {
                "texts": texts,
                "mode": _mode_to_api_value(mode_label),
                "batch_size": min(len(texts), 16),
            },
        )
        rows = [
            [item["original"], item["corrected"], item["status"]]
            for item in response["results"]
        ]
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            newline="",
            encoding="utf-8",
        )
        with temp_file as handle:
            writer = csv.writer(handle)
            writer.writerow(["Original", "Corrected", "Status"])
            writer.writerows(rows)
        return rows, temp_file.name

    def detect_ui(text: str) -> tuple[list[tuple[str, str | None]], dict[str, Any]]:
        """Submit an error-detection request."""

        try:
            response = client.post("/detect", {"text": text})
            return _highlight_text(text, response["errors"]), response["errors"]
        except Exception as exc:  # pragma: no cover - UI error path
            error_payload = _safe_json_error(exc)
            return [(text, "Error")], error_payload

    def add_rules_ui(rules_text: str) -> Dict[str, Any]:
        """Add newline-separated rules to the knowledge base."""

        rules = [line.strip() for line in rules_text.splitlines() if line.strip()]
        return client.post("/knowledge/add", {"rules": rules})

    def search_rules_ui(query: str, top_k: int) -> Dict[str, Any]:
        """Search the knowledge base for relevant rules."""

        return client.get("/knowledge/search", params={"query": query, "top_k": top_k})

    def list_prompts_ui() -> tuple[list[str], str, list[list[Any]]]:
        """Fetch prompt registry contents for the prompt-management tab."""

        response = client.get("/prompts")
        versions = response["versions"]
        table = [
            [
                item["version_id"],
                item["description"],
                item["is_active"],
                item["metrics"],
            ]
            for item in versions
        ]
        return (
            [item["version_id"] for item in versions],
            response["active_version"],
            table,
        )

    def view_prompt_ui(version_id: str) -> str:
        """Fetch and display a single prompt template."""

        if not version_id:
            return ""
        response = client.get(f"/prompts/{version_id}")
        return response["template"]

    def promote_prompt_ui(
        version_id: str,
    ) -> tuple[dict[str, Any], list[str], str, list[list[Any]]]:
        """Promote a prompt and refresh prompt-management widgets."""

        response = client.post(f"/prompts/{version_id}/promote", {})
        versions, active_version, table = list_prompts_ui()
        return response, versions, active_version, table

    def rollback_prompt_ui() -> tuple[dict[str, Any], list[str], str, list[list[Any]]]:
        """Rollback the active prompt and refresh prompt-management widgets."""

        response = client.post("/prompts/rollback", {})
        versions, active_version, table = list_prompts_ui()
        return response, versions, active_version, table

    def evaluate_ui(file_obj: Any, metric_names: List[str]) -> tuple[Any, str | None]:
        """Upload prediction/reference CSV data and render evaluation metrics."""

        if file_obj is None:
            return None, None

        file_path = Path(getattr(file_obj, "name", file_obj))
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            predictions: List[str] = []
            references: List[str] = []
            for row in reader:
                prediction = row.get("prediction") or row.get("predictions")
                reference = row.get("reference") or row.get("references")
                if prediction and reference:
                    predictions.append(prediction)
                    references.append(reference)

        response = client.post(
            "/evaluate",
            {
                "predictions": predictions,
                "references": references,
                "metrics": metric_names or ["exact_match"],
            },
        )
        plot = _build_metric_plot(response["metrics"])
        report_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        )
        with report_file as handle:
            json.dump(response, handle, indent=2)
        return plot, report_file.name

    css = """
    .app-shell {max-width: 1120px; margin: 0 auto;}
    .hero-title {font-size: 2.4rem; font-weight: 700; letter-spacing: -0.03em;}
    .hero-subtitle {color: #4b5563; margin-bottom: 1rem;}
    """

    with gr.Blocks(title="Grammar Autocorrector") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(f"<style>{css}</style>")
            gr.Markdown(
                "<div class='hero-title'>Grammar Autocorrector</div>",
            )
            gr.Markdown(
                "<div class='hero-subtitle'>Powered by T5, BERT, and a RAG pipeline.</div>",
            )

            with gr.Tabs():
                with gr.Tab("Correct Text"):
                    input_text = gr.Textbox(
                        label="Enter text",
                        lines=4,
                        max_lines=10,
                        value=examples[0],
                    )
                    mode = gr.Radio(
                        ["Auto", "T5 Only", "RAG-Enhanced"],
                        value="Auto",
                        label="Mode",
                    )
                    num_beams = gr.Slider(
                        minimum=2,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Number of beams",
                    )
                    show_errors = gr.Checkbox(
                        label="Show detected errors",
                        value=True,
                    )
                    correct_button = gr.Button("Correct", variant="primary")
                    corrected_text = gr.Textbox(label="Corrected text", lines=4)
                    error_highlights = gr.HighlightedText(label="Detected errors")
                    guardrail_status = gr.JSON(label="Guardrail report")
                    processing_time = gr.Textbox(label="Processing time")
                    gr.Examples(examples=examples[:5], inputs=[input_text])
                    correct_button.click(
                        correct_text_ui,
                        inputs=[input_text, mode, num_beams, show_errors],
                        outputs=[
                            corrected_text,
                            error_highlights,
                            guardrail_status,
                            processing_time,
                        ],
                    )

                with gr.Tab("Batch Correct"):
                    batch_input = gr.Dataframe(
                        headers=["Input Text"],
                        row_count=5,
                        column_count=(1, "fixed"),
                        label="Input texts",
                    )
                    batch_mode = gr.Radio(
                        ["Auto", "T5 Only", "RAG-Enhanced"],
                        value="Auto",
                        label="Mode",
                    )
                    batch_button = gr.Button("Process batch", variant="primary")
                    batch_output = gr.Dataframe(
                        headers=["Original", "Corrected", "Status"],
                        label="Batch results",
                    )
                    batch_file = gr.File(label="Download results")
                    batch_button.click(
                        correct_batch_ui,
                        inputs=[batch_input, batch_mode],
                        outputs=[batch_output, batch_file],
                    )

                with gr.Tab("Detect Errors"):
                    detect_input = gr.Textbox(label="Enter text", lines=4)
                    detect_button = gr.Button("Detect", variant="primary")
                    detect_output = gr.HighlightedText(label="Detected errors")
                    detect_json = gr.JSON(label="Error details")
                    detect_button.click(
                        detect_ui,
                        inputs=[detect_input],
                        outputs=[detect_output, detect_json],
                    )

                with gr.Tab("Knowledge Base"):
                    kb_rules = gr.Textbox(
                        label="Add rules (one per line)",
                        lines=6,
                    )
                    kb_add = gr.Button("Add rules", variant="primary")
                    kb_add_result = gr.JSON(label="Add result")
                    kb_query = gr.Textbox(label="Search query")
                    kb_top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Top K",
                    )
                    kb_search = gr.Button("Search", variant="primary")
                    kb_search_result = gr.JSON(label="Search results")
                    kb_add.click(
                        add_rules_ui, inputs=[kb_rules], outputs=[kb_add_result]
                    )
                    kb_search.click(
                        search_rules_ui,
                        inputs=[kb_query, kb_top_k],
                        outputs=[kb_search_result],
                    )

                with gr.Tab("Prompt Manager"):
                    prompt_choices = gr.Dropdown(
                        label="Prompt versions",
                        choices=[],
                        value=None,
                    )
                    active_version = gr.Textbox(label="Active version")
                    prompt_template = gr.Code(
                        label="Prompt template",
                        language="markdown",
                    )
                    prompt_history = gr.Dataframe(
                        headers=["Version", "Description", "Active", "Metrics"],
                        label="Version history",
                    )
                    prompt_status = gr.JSON(label="Action result")
                    prompt_promote = gr.Button("Promote", variant="primary")
                    prompt_rollback = gr.Button("Rollback")
                    refresh_button = gr.Button("Refresh")

                    refresh_button.click(
                        list_prompts_ui,
                        outputs=[prompt_choices, active_version, prompt_history],
                    )
                    prompt_choices.change(
                        view_prompt_ui,
                        inputs=[prompt_choices],
                        outputs=[prompt_template],
                    )
                    prompt_promote.click(
                        promote_prompt_ui,
                        inputs=[prompt_choices],
                        outputs=[
                            prompt_status,
                            prompt_choices,
                            active_version,
                            prompt_history,
                        ],
                    )
                    prompt_rollback.click(
                        rollback_prompt_ui,
                        outputs=[
                            prompt_status,
                            prompt_choices,
                            active_version,
                            prompt_history,
                        ],
                    )

                with gr.Tab("Evaluate"):
                    eval_file = gr.File(
                        label="Upload CSV with prediction/reference columns"
                    )
                    eval_metrics = gr.CheckboxGroup(
                        ["gleu", "rouge", "exact_match"],
                        value=["gleu", "rouge", "exact_match"],
                        label="Metrics",
                    )
                    eval_button = gr.Button("Run evaluation", variant="primary")
                    eval_plot = gr.Plot(label="Metrics plot")
                    eval_report = gr.File(label="Download evaluation report")
                    eval_button.click(
                        evaluate_ui,
                        inputs=[eval_file, eval_metrics],
                        outputs=[eval_plot, eval_report],
                    )

            gr.Markdown("---")
            gr.Markdown(
                "**Author:** Parva Barot | **Model:** T5-base fine-tuned | "
                "**Version:** 1.0.0"
            )

            demo.load(
                list_prompts_ui,
                outputs=[prompt_choices, active_version, prompt_history],
            )
    return demo


if __name__ == "__main__":
    application = build_app()
    application.launch(server_name="0.0.0.0", server_port=7860, share=False)

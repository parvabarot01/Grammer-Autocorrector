# Known Issues

- Model weights are not included in the repository. Download or train checkpoints locally before expecting true T5 or BERT model behavior.
- The toxicity filter in v1 is keyword-based rather than classifier-based, so it prioritizes latency and transparency over nuance.
- BERT fine-tuning requires token-level error labels. If you do not have them, generate weak labels or synthetic alignment labels first.
- The RAG pipeline requires knowledge-base initialization before first use. The API boot process handles this automatically, but standalone scripts should call `load_all()` or build/load the vector store explicitly.
- The public Next.js frontend is an MVP. Login is not implemented yet.
- An admin dashboard is not included in Sprint 9. Continue using the optional legacy Gradio UI for operator workflows.
- The public `POST /public/correct` response is intentionally simplified and does not expose internal model, retrieval, evaluation, dataset, or guardrail details.
- Cloud deployment is not configured yet. Sprint 9 supports local and Docker Compose workflows.

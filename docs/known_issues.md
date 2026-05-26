# Known Issues

- Model weights are not included in the repository. Download or train checkpoints locally before expecting true T5 or BERT model behavior.
- The toxicity filter in v1 is keyword-based rather than classifier-based, so it prioritizes latency and transparency over nuance.
- BERT fine-tuning requires token-level error labels. If you do not have them, generate weak labels or synthetic alignment labels first.
- The RAG pipeline requires knowledge-base initialization before first use. The API boot process handles this automatically, but standalone scripts should call `load_all()` or build/load the vector store explicitly.

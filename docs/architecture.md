# Architecture Decision Records

## ADR-001: Choice of T5 Over GPT-2 for Sequence-to-Sequence Grammar Correction

### Status
Accepted

### Context
The project requires a model that can take erroneous input text and generate corrected output text reliably. Grammar correction is inherently a sequence-to-sequence rewriting task.

### Decision
Use T5 as the primary grammar correction model instead of GPT-2.

### Consequences

- T5 aligns naturally with text-to-text transformation tasks.
- Fine-tuning workflows for correction datasets are well established.
- Decoder-only GPT-2 would require more prompt engineering and less direct task alignment.
- T5 introduces larger inference cost than simpler baselines but improves design clarity and likely correction quality.

## ADR-002: BERT for Error Detection vs. Rule-Based Approaches

### Status
Accepted

### Context
The system needs token-level error detection to explain mistakes, support auto-routing, and enrich UI feedback. Rule-based detection is brittle across sentence patterns and error types.

### Decision
Use BERT token classification for grammar error detection rather than relying on handcrafted grammar rules.

### Consequences

- Detection quality can generalize better across varied constructions.
- The architecture gains a learnable, explainable detector component.
- Training data preparation is more complex than authoring simple rule sets.
- Some deterministic grammar rules may still be useful as supplemental heuristics.

## ADR-003: RAG Pipeline Architecture Using FAISS Vector Store

### Status
Accepted

### Context
The project includes a context-aware correction path that retrieves grammar rules or reference examples. The system should be affordable, local-development friendly, and easy to version.

### Decision
Use a FAISS-backed RAG pipeline with sentence embeddings and file-based persistence.

### Consequences

- Local experimentation remains cheap and offline-friendly.
- Retrieval quality can be iterated independently from core correction models.
- File-based persistence is simpler than managed vector infrastructure but less scalable for multi-tenant cloud workloads.

## ADR-004: FastAPI Over Flask for Serving Inference Endpoints

### Status
Accepted

### Context
The service requires modern validation, strong OpenAPI generation, async-friendly patterns, and typed request/response models.

### Decision
Use FastAPI as the primary backend framework instead of Flask.

### Consequences

- Native Pydantic integration improves API contract clarity.
- OpenAPI docs are available with less custom code.
- FastAPI supports future async extensions more cleanly.
- The team accepts an additional dependency and framework conventions in exchange for better developer ergonomics.

## ADR-005: Gradio for Rapid UI Prototyping Over a Custom React Frontend

### Status
Accepted

### Context
The project needs a user-friendly demonstration and validation interface, but early development time should prioritize NLP capabilities rather than custom frontend engineering.

### Decision
Use Gradio for the first-generation UI instead of building a custom React application.

### Consequences

- Development speed is significantly improved.
- UI components for text input, tables, and visualization are available quickly.
- Long-term visual customization is more limited than a custom frontend.
- A future React frontend can still be added after core workflows stabilize.

## ADR-006: Choice of FAISS Over Pinecone for the Initial Vector Store

### Status
Accepted

### Context
The RAG subsystem needs a vector index that works for local development, offline experiments, and inexpensive iterative testing. Managed vector databases would add operational cost and external service dependencies too early in the project.

### Decision
Use FAISS as the primary vector-store backend for the first release instead of Pinecone.

### Consequences

- The project can run retrieval locally without external infrastructure.
- Development and testing stay inexpensive and portable.
- Persistence is file-based and simple to version conceptually.
- Scaling to hosted, multi-tenant use cases will require a later migration path or adapter layer.

## ADR-007: Rule-Based Toxicity Filter vs. ML Classifier

### Status
Accepted

### Context
The project needs responsible AI guardrails, but the early product goal emphasizes low latency, straightforward explainability, and minimal infrastructure overhead. A full toxicity model would improve nuance but would also add dependencies and serving complexity.

### Decision
Use a rule-based toxicity filter in v1 instead of a dedicated ML toxicity classifier.

### Consequences

- Guardrail checks remain fast and easy to inspect.
- False positives and false negatives are more likely than with a tuned classifier.
- The implementation is easier to test and deploy in offline or low-resource settings.
- A later release can swap the rule-based layer for a classifier behind the same interface if quality requirements increase.

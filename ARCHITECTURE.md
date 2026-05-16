# Lit-Agent: Technical Architecture & Design Document

**Autonomous Research Hypothesis Generator**
Powered by NVIDIA NeMoClaw · Deployed on Brev · Backend Engineering Prototype

---

## 1. Executive Summary

Lit-Agent is a backend-only autonomous research assistant that accepts a research topic, retrieves relevant literature context, and uses NVIDIA NeMoClaw to synthesise a structured experiment hypothesis — all without human intervention after the initial request.

The system solves a specific, bounded problem: the ideation bottleneck in early-stage research. Literature surveys are repetitive, time-consuming, and structurally similar enough that an LLM-powered pipeline can automate the synthesis step. Rather than building a chatbot, Lit-Agent is designed as a **pipeline system** — each request triggers a deterministic multi-stage workflow that transitions through explicit states, persists its output to a relational database, and exposes results through a REST API.

NeMoClaw serves as the inference and structured-output enforcement layer. Instead of making raw API calls and parsing freeform text, NeMoClaw is used to bind the LLM's response to a strict Pydantic schema, making the hypothesis output machine-readable and directly insertible into the database. This is the architectural distinction between a demo chatbot and an engineered AI pipeline.
  
**What the demo showcases:**
- Asynchronous task orchestration with a visible state machine
- NeMoClaw-enforced structured outputs persisted to SQLite
- A clean REST API suitable for frontend or downstream service consumption
- Backend engineering patterns (session management, dependency injection, audit logging) applied to an AI workflow

---

## 2. Problem Statement

### The Inefficiency of Manual Literature Review

A researcher scoping a new project typically spends several hours reading paper abstracts before identifying a viable research gap. This process is structurally repetitive: read abstracts, identify limitations, cross-reference gaps, propose a direction. The cognitive load is real, but the *pattern* of work is mechanical enough to be automated.

Existing tooling addresses fragments of this problem. Semantic Scholar and Elicit provide search and summarisation. But neither produces a structured, database-persisted hypothesis that a downstream system can act on. The output is always human-readable prose, not a machine-consumable record.

### Why Autonomous Orchestration Helps

The core argument for an automated pipeline is not that the LLM replaces the researcher — it is that it compresses the *initial ideation phase* from hours to seconds. A researcher can submit five topics, review five structured hypotheses, and discard four in five minutes rather than five hours. The AI acts as a first-pass filter, not a final authority.

This pattern — autonomous first-pass analysis followed by human review — is precisely where backend orchestration adds value. The system needs to be reliable, traceable, and fast. It needs to produce consistent output schemas so results are comparable across topics. It needs to log what it did so researchers can understand why a hypothesis was generated. None of these requirements are satisfied by a chat interface.

### The Backend Engineering Problem

Building this correctly requires solving several non-trivial backend problems simultaneously: decoupling long-running LLM inference from synchronous HTTP responses, enforcing structured outputs from probabilistic models, managing database state across asynchronous threads, and providing a clean API contract that is stable regardless of what happens inside the pipeline. These are the engineering challenges this project addresses.

---

## 3. System Goals

### Primary Goals

| Goal | Rationale |
|---|---|
| Rapid research synthesis | Compress literature review from hours to seconds |
| Structured hypothesis generation | Machine-readable output, not freeform prose |
| Lightweight autonomous reasoning | Multi-stage pipeline, no human intervention |
| Modular backend architecture | Each component has a single responsibility |
| API-first design | Backend is consumption-ready by any client |
| Persistence and traceability | Every run is stored; every state transition is logged |
| NeMoClaw integration | Inference + structured output enforcement via NVIDIA stack |

### Intentional Exclusions (MVP Scope)

Several capabilities were deliberately excluded to keep the scope achievable in a 3–4 hour build window:

- **Real PDF/arXiv ingestion** — scraping and parsing academic PDFs introduces significant error surface. The literature retrieval layer is mocked with realistic, curated abstracts. The *architectural slot* for a real retrieval service exists and is clearly labelled.
- **Vector database / semantic search** — a production system would embed paper abstracts and retrieve by cosine similarity. For this prototype, retrieval is keyword-based. Swapping in a Milvus or Qdrant client requires changing one function in `mock_data.py`.
- **Multi-agent coordination** — the pipeline is single-threaded and sequential. Parallel agent execution (planner, summariser, and critic running concurrently) is a natural extension but adds coordination complexity that is out of scope.
- **Frontend** — the Swagger UI auto-generated by FastAPI serves as an interactive demo interface. A frontend is architecturally straightforward to add but adds no value for demonstrating backend engineering capability.
- **Authentication** — excluded for demo velocity. Production would add OAuth2 or API key middleware at the FastAPI layer.

---

## 4. High-Level System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
│          (curl / Swagger UI / future Next.js frontend)              │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FASTAPI ROUTER LAYER                           │
│   POST /analyze  GET /task/{id}  GET /task/{id}/results  /logs     │
│                                                                     │
│   • Input validation (Pydantic)                                     │
│   • Task creation + DB write                                        │
│   • BackgroundTask dispatch                                         │
│   • Response serialisation                                          │
└────────────┬────────────────────────────────────┬───────────────────┘
             │ 202 Accepted                        │ Background thread
             │ (immediate)                         ▼
             │                    ┌────────────────────────────────────┐
             │                    │        AGENT RUNNER                │
             │                    │      (State Machine)               │
             │                    │                                    │
             │                    │  PENDING → RETRIEVING              │
             │                    │         → SYNTHESIZING             │
             │                    │         → COMPLETED | FAILED       │
             │                    │                                    │
             │                    │  • Owns all status mutations       │
             │                    │  • Writes audit log entries        │
             │                    │  • Handles all exceptions          │
             │                    └────────┬──────────────┬────────────┘
             │                             │              │
             │                             ▼              ▼
             │              ┌──────────────────┐  ┌───────────────────┐
             │              │  MOCK DATA LAYER │  │  NEMOCLAW CLIENT  │
             │              │                  │  │                   │
             │              │  4 topic corpora │  │  Prompt compiler  │
             │              │  3–4 abstracts   │  │  NIM API caller   │
             │              │  Keyword routing │  │  Schema enforcer  │
             │              │                  │  │  Retry handler    │
             │              └──────────────────┘  └────────┬──────────┘
             │                                             │
             │                                             ▼
             │                                  ┌──────────────────────┐
             │                                  │   NVIDIA NIM API     │
             │                                  │  (LLaMA 3.1 70B)     │
             │                                  │  /chat/completions   │
             │                                  └──────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PERSISTENCE LAYER (SQLite)                      │
│                                                                     │
│   research_tasks         generated_hypotheses    agent_run_logs     │
│   ─────────────          ──────────────────────  ───────────────    │
│   id (PK/UUID)           id (PK)                 id (PK)           │
│   topic                  task_id (FK)             task_id (FK)     │
│   status                 gap_identified           stage             │
│   error_message          proposed_architecture    message           │
│   created_at             evaluation_metric        timestamp         │
│   updated_at             confidence_score                           │
│                          raw_llm_output                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Request Lifecycle

1. Client sends `POST /api/v1/research/analyze` with `{"topic": "..."}`.
2. FastAPI validates the request body against `AnalyzeTopicRequest` (Pydantic). Rejects malformed input before any DB write.
3. A `ResearchTask` row is created in SQLite with `status=PENDING`. The `task_id` UUID is returned immediately with HTTP 202.
4. FastAPI's `BackgroundTasks` dispatches `_run_in_background()` on a worker thread. A **new** `SessionLocal()` is created for this thread — the request-scoped session is intentionally not shared across thread boundaries.
5. `AgentRunner.run_research_pipeline()` executes the state machine, mutating status at each stage and flushing to DB so polls return live state.
6. NeMoClaw is invoked during the `SYNTHESIZING` stage. On success, a `GeneratedHypothesis` row is written and status moves to `COMPLETED`. On any exception, status moves to `FAILED` with an error message stored.
7. Client polls `GET /task/{id}` until `COMPLETED`, then retrieves the hypothesis via `GET /task/{id}/results`.

### Why This Request Pattern?

The 202 + polling pattern was chosen over WebSockets or Server-Sent Events deliberately. LLM inference can take 5–30 seconds depending on the model and load. A synchronous HTTP request held open for that duration is fragile — load balancers time out, clients disconnect, and the server holds a thread. The decoupled pattern is standard practice for any long-running computation exposed via HTTP.

Polling also makes the state transitions *visible*, which is architecturally valuable for a demo: you can watch the status field advance in real time.

---

## 5. Autonomous Workflow Design

### The Pipeline as a Lightweight AI Scientist

The system is structured as a sequential four-stage pipeline that simulates a bounded version of what a human researcher does during early ideation:

```
Stage 1: RETRIEVE
  Input:  research topic (string)
  Action: keyword-match topic to corpus, extract 3–4 paper abstracts
  Output: structured context block (text)

Stage 2: SYNTHESIZE
  Input:  context block + system prompt
  Action: NeMoClaw compiles prompt, submits to NIM, enforces schema
  Output: HypothesisOutput (gap, architecture, metric, confidence)

Stage 3: VALIDATE
  Action: Pydantic validates the LLM response against HypothesisOutput schema
  Output: validated Python object or exception (triggers retry)

Stage 4: PERSIST
  Action: Write GeneratedHypothesis row, update ResearchTask to COMPLETED
  Output: structured hypothesis available via GET /results
```

Each stage is a separate logical responsibility. A failure at any stage sets `status=FAILED` and stores the error — the system never hangs silently.

### Why Sequential Rather Than Parallel?

A multi-agent swarm (planner + summariser + critic running in parallel) would be more impressive on paper but would add coordination overhead — shared state, merge logic, conflict resolution — that would consume the majority of the build time without meaningfully improving the demo output. The sequential design is not a simplification of ambition; it is an accurate reflection of how production pipelines are usually structured when the bottleneck is a single LLM call.

The pipeline stages are cleanly separable. Each reads from the previous stage's output and writes to the next stage's input. This makes the system easy to extend: inserting a `CRITIC` stage between `SYNTHESIZE` and `PERSIST` (for a second LLM pass that evaluates the hypothesis quality) requires adding ~20 lines to `agent_runner.py` and a new status value.

### How Structured Reasoning Is Enforced

The LLM is not asked to reason freely. The system prompt instructs it to produce **only** a JSON object matching a specific schema. The user prompt injects the literature context and reinforces the schema. If the LLM returns anything else — prose, markdown fences, explanations — the `NeMoClawClient._parse_and_validate()` method strips the wrapping and validates the extracted JSON against `HypothesisOutput`. If validation fails, the client retries up to two times before raising `NeMoClawError`.

This is the key insight behind pipeline-oriented AI systems: you do not trust the model's output format. You enforce it at the application layer.

### Task Persistence as Workflow Memory

Every pipeline run is independently persisted. This means:

- Researchers can compare hypotheses across different topics side-by-side by querying `GET /tasks`.
- Failed runs retain their error messages, making debugging tractable.
- The audit log (`agent_run_logs`) records every state transition with a timestamp and message, providing a complete execution trace for any run.

In a production system, this persistence layer would support iterative workflows: a researcher could re-run a topic with a different model or prompt configuration and compare results against previous runs.

---

## 6. NeMoClaw Integration Architecture

### Why NeMoClaw?

NeMoClaw is positioned as NVIDIA's orchestration runtime for LLM-powered agent workflows, sitting above raw inference and below full application logic. Its value proposition in this architecture is twofold:

1. **Inference against NIM**: NeMoClaw targets NVIDIA's NIM (NVIDIA Inference Microservices) endpoints, which serve optimised quantised versions of frontier models on NVIDIA hardware. On Brev, the same code that calls the hosted NIM API can be redirected to a local NIM container by changing one environment variable (`NIM_BASE_URL`).

2. **Structured output contract**: Rather than prompting the LLM and parsing freeform text, NeMoClaw's design aligns with enforcing typed schemas on model outputs. In this implementation, `HypothesisOutput` (a Pydantic model) serves as the response contract — the client layer validates the raw LLM string against this schema before any downstream processing.

### Orchestration Workflow

```
AgentRunner
    │
    │  topic + papers context
    ▼
NeMoClawClient.run(
    system_prompt = SYNTHESIS_SYSTEM_PROMPT,  ← prompt template module
    user_prompt   = SYNTHESIS_USER_TEMPLATE.format(topic, context),
    response_model = HypothesisOutput         ← Pydantic schema contract
)
    │
    │  compile messages list
    ▼
NeMoClawClient._call_nim()
    │
    │  POST {base_url}/chat/completions
    │  Authorization: Bearer {NVIDIA_API_KEY}
    │  model: meta/llama-3.1-70b-instruct
    ▼
NVIDIA NIM Endpoint
    │
    │  raw assistant message (JSON string)
    ▼
NeMoClawClient._parse_and_validate()
    │
    │  strip markdown fences → json.loads() → HypothesisOutput.model_validate()
    ▼
(HypothesisOutput, raw_json_str)
    │
    ▼
AgentRunner → persist to GeneratedHypothesis table
```

### Prompt Architecture

Prompts are isolated in `app/prompts/research_synthesis.py`. This is not incidental — prompt engineering is a distinct concern from orchestration logic, and mixing them creates maintenance problems. Changing the system prompt to improve hypothesis quality should not require touching the agent runner.

The system prompt establishes the persona and strict JSON output rule. The user template injects the dynamic variables (topic and context block). The NeMoClaw client compiles these into the `messages` array for the NIM API call.

### Handling the Real vs. Mocked Boundary

The `NeMoClawClient` and `MockNeMoClawClient` share an identical interface. The factory function `get_nemoclaw_client()` selects between them based on whether `NVIDIA_API_KEY` is present in the environment. The rest of the system has no awareness of which client is active — this is standard dependency inversion applied to an AI inference layer.

**What is real:** The NIM API integration (`_call_nim()`), the prompt templates, the schema validation, the retry logic, the audit logging, the full DB persistence pipeline.

**What is mocked:** The literature retrieval layer (keyword lookup instead of arXiv/Semantic Scholar), and the mock client's LLM responses (deterministic, realistic, but not live inference).

This distinction is intentional and honest. In a live demo with an `NVIDIA_API_KEY` set, the mock client is never invoked — the full NIM inference path runs. The mock exists purely to make the system demoable offline.

### Future NeMoClaw Extensibility

The client abstraction supports a clean upgrade path as NeMoClaw's SDK matures:

- **Tool use / function calling**: NeMoClaw supports binding tools to agents. The retrieval layer (currently mocked) could be registered as a NeMoClaw tool, making the literature search genuinely agentic rather than procedurally injected.
- **Guardrails**: NeMo Guardrails can be layered onto the NeMoClaw runtime to reject off-topic inputs or enforce content policies without modifying application logic.
- **Multi-model routing**: The client's `_model` field is a configuration parameter. Swapping `meta/llama-3.1-70b-instruct` for a domain-specific model (e.g., a biomedical fine-tune) requires changing one line in `.env`.

---

## 7. API Design Philosophy

### Why REST Over GraphQL or gRPC?

REST was chosen for three practical reasons: universal client compatibility (curl, Swagger UI, any HTTP library), alignment with standard webhook and polling patterns, and zero additional tooling or schema compilation overhead. GraphQL would be appropriate if clients needed fine-grained field selection; gRPC would be appropriate for high-throughput inter-service communication. Neither applies here.

### Stateless Endpoints, Stateful Resources

Each endpoint is stateless in the HTTP sense — no session cookies, no server-side client state. All application state lives in the database, addressed by `task_id`. This makes the API trivially scalable: multiple FastAPI workers can serve requests concurrently because they all read from and write to the same SQLite file (or, in a production deployment, the same PostgreSQL instance).

### Async Task Pattern

```
POST /analyze          → 202 Accepted + task_id     (immediate)
GET  /task/{id}        → current status              (poll until COMPLETED)
GET  /task/{id}/results → structured hypothesis      (fetch once COMPLETED)
GET  /task/{id}/logs   → full execution audit trail  (available any time)
```

This pattern decouples the client's request from the pipeline's execution time. It also makes the system's internal behaviour observable — the status transitions are externally visible, which is both useful for debugging and compelling in a live demo.

### Response Schema Consistency

Every endpoint returns a typed Pydantic model. This ensures that the response shape is documented, validated, and predictable. `TaskStatusResponse`, `TaskResultResponse`, `HypothesisResponse` — each schema has a single responsibility and maps cleanly to a database query. There are no "catch-all" response dictionaries.

### Example Interaction

```
→ POST /api/v1/research/analyze
  Body: {"topic": "efficient llm routing strategies"}
← 202 {"task_id": "abc-123", "status": "PENDING", "message": "..."}

→ GET /api/v1/research/task/abc-123
← 200 {"task_id": "abc-123", "status": "SYNTHESIZING", ...}

→ GET /api/v1/research/task/abc-123/results
← 200 {
    "task_id": "abc-123",
    "status": "COMPLETED",
    "hypothesis": {
      "gap_identified": "Routing decisions ignore inter-token semantic continuity...",
      "proposed_architecture": "SeqRouter: a sequence-level routing transformer...",
      "evaluation_metric": "MMLU accuracy degradation vs. routing latency on MT-Bench",
      "confidence_score": "HIGH"
    }
  }
```

---

## 8. Database & Persistence Design

### Why Persistence Matters in Autonomous Systems

An autonomous pipeline without persistence is a black box. You submit a request and get a result, but you cannot answer: What did the agent actually do? What intermediate states did it pass through? What was the raw model output before parsing? Why did this run fail but the previous one succeed?

The persistence design in Lit-Agent is specifically structured to answer these questions.

### Schema Rationale

**`research_tasks`** is the control record. It owns the lifecycle state machine. Its `status` field is the single source of truth for where the pipeline is. Its `error_message` field captures failures without requiring a log scrape. Its `updated_at` field enables elapsed-time calculations.

**`generated_hypotheses`** is the output record. It is in a separate table (not columns on `research_tasks`) for a specific reason: the hypothesis row should not exist until the pipeline completes successfully. A missing row is a meaningful signal — it means the task has not yet produced output, which is different from a task that produced an empty output. One-to-one FK with cascade delete ensures no orphaned hypothesis rows.

**`agent_run_logs`** is the audit record. Every state transition appends a row. This is append-only by design — logs are never updated or deleted (only when their parent task is deleted via cascade). The log table answers "what did the agent do and in what order?" without requiring log aggregation infrastructure.

### SQLite vs PostgreSQL

SQLite was chosen for the MVP because it requires zero infrastructure setup — the database is a single file that is created at startup. This was the correct choice for a 3–4 hour prototype.

The tradeoff is clear: SQLite has write serialisation (only one writer at a time). For this workload — one background task writing per pipeline run, multiple readers polling — this is not a constraint. A production system with multiple concurrent pipeline runs would hit write contention and would require migrating to PostgreSQL. Because SQLAlchemy's ORM abstracts the database entirely, this migration involves changing the `DATABASE_URL` environment variable and installing `psycopg2`. No application code changes.

### `raw_llm_output` Field

The `GeneratedHypothesis` table stores the raw JSON string returned by the LLM alongside the parsed fields. This is intentional. In a probabilistic system, storing only the parsed output discards information about how the model arrived at the answer. Retaining the raw output enables post-hoc analysis: you can compare the raw outputs of two runs on the same topic to understand whether the model's reasoning changed, even if the structured fields look identical.

---

## 9. Engineering Tradeoffs

### Simplicity vs. Scalability

The system is explicitly not designed for horizontal scalability in its current form. SQLite, in-process background tasks, and a single uvicorn worker are sufficient for a demo workload of tens of requests. A production system serving thousands of concurrent users would require:

- PostgreSQL with connection pooling
- A task queue (Celery + Redis or Dramatiq) to decouple pipeline execution from the API process
- Multiple worker instances behind a load balancer
- Distributed tracing (OpenTelemetry) instead of DB-backed audit logs

None of these were implemented because they would have consumed the entire build time without improving the demo's ability to communicate the core architectural ideas. The architecture is designed to *accommodate* these additions — not to *require* them at MVP stage.

### Modularity vs. Implementation Speed

The codebase is more modular than strictly necessary for a 3–4 hour project. Separate modules for prompts, mock data, the NeMoClaw client, the agent runner, schemas, and routes — rather than a single `main.py` — added ~30 minutes of scaffolding time. This was a deliberate investment: the modular structure communicates architectural intent more clearly than a monolithic file, and it makes the codebase defensible in a code review.

### Realism vs. Feasibility

The literature retrieval layer is the most significant compromise. Real arXiv scraping and PDF parsing would have made the demo meaningfully more impressive, but would have introduced error surface (network failures, PDF encoding issues, rate limits) that would be difficult to handle gracefully in the time available. The mock data is written to be realistic enough that the overall system behavior is authentic — the LLM receives the same quality of context it would receive from a real retrieval system.

### Orchestration Depth vs. Time Constraints

The pipeline could have included a `CRITIC` stage — a second LLM pass that evaluates the hypothesis for feasibility before persisting it. This would have made the system more sophisticated without adding infrastructure complexity. It was excluded because getting the core pipeline (retrieve → synthesise → persist) reliable and well-tested was the higher priority.

### Frontend Exclusion

Excluding a frontend was not a limitation — it was an architectural decision. The Swagger UI auto-generated by FastAPI (`/docs`) provides a fully functional interactive interface for the demo. Building a React frontend would have shifted focus from backend engineering to UI/UX, which is not the capability being demonstrated. Any frontend that consumes a well-defined REST API can be built independently and attached later.

---

## 10. Brev Deployment Architecture

### Why Brev?

Brev provides pre-configured Ubuntu instances with GPU access and a simple port-forwarding workflow. For a project that needs to demonstrate NIM inference (which benefits from GPU acceleration) without requiring local NVIDIA hardware, Brev is the appropriate deployment target. The alternative — running everything locally on a CPU — works with the mock client but cannot demonstrate live NIM inference.

### Environment Architecture on Brev

```
Brev Instance (Ubuntu 22.04, T4 GPU)
├── Python 3.12 venv
├── FastAPI + uvicorn (port 8000)
├── SQLite database (lit_agent.db, local file)
├── .env (NVIDIA_API_KEY, NIM_BASE_URL, NIM_MODEL)
└── NIM connectivity: NVIDIA hosted API or local NIM container

Local Machine
└── brev port-forward <instance> 8000:8000
    → http://localhost:8000/docs  (Swagger UI in browser)
```

### GPU Considerations

The FastAPI application itself is CPU-only — it makes HTTP calls to the NIM inference endpoint and performs DB I/O. GPU is only relevant if running a local NIM container on the Brev instance (e.g., `nvidia/nim/meta/llama-3.1-70b-instruct`). For this prototype, the hosted NIM API is used, which means GPU on the Brev instance is not strictly required. A standard CPU instance (or even a local machine) runs the full system, with inference offloaded to NVIDIA's infrastructure.

### Deployment Sequence

```bash
# On Brev instance
git clone <repo> && cd demo/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in NVIDIA_API_KEY
uvicorn main:app --host 0.0.0.0 --port 8000

# On local machine
brev port-forward <machine-name> 8000:8000
# Open: http://localhost:8000/docs
```

### Operational Simplicity

The absence of Docker, Kubernetes, or a process manager (e.g., supervisord) is intentional for the demo context. Adding a `Dockerfile` is a 10-line exercise; the application is already configured with `--host 0.0.0.0` for container compatibility. The intentional choice was to keep the operational complexity at the level appropriate for a prototype evaluation, not a production deployment.

---

## 11. Future Improvements

The architecture was designed with extension points in mind. The following improvements are realistic next steps, ordered by implementation complexity:

### Near-Term (Low Complexity)

| Improvement | Mechanism | Impact |
|---|---|---|
| Real arXiv retrieval | Replace `mock_data.py` with `arxiv` Python client | Real literature grounding |
| Critic/evaluator stage | Add `EVALUATING` state, second NeMoClaw call | Hypothesis quality scoring |
| Topic search history | Add `GET /tasks?topic=...` query param | Research session continuity |
| PostgreSQL migration | Change `DATABASE_URL`, add `psycopg2` | Production-grade persistence |

### Medium-Term (Moderate Complexity)

| Improvement | Mechanism | Impact |
|---|---|---|
| Vector retrieval (RAG) | Qdrant/Milvus + embedding model via NIM | Semantic literature matching |
| Multi-topic comparison | Batch endpoint, hypothesis ranking | Research portfolio view |
| NeMo Guardrails | Layer onto NeMoClaw client | Input/output safety enforcement |
| Task queue (Celery) | Replace `BackgroundTasks` | Scalable concurrent pipelines |

### Long-Term (Architectural Changes)

- **Multi-agent coordination**: Separate planner, summariser, and hypothesis-generator agents communicating via a message bus, enabling parallel execution and specialisation.
- **Experiment graph**: Persist relationships between hypotheses (e.g., "Hypothesis B extends Hypothesis A"). Enables research lineage tracking.
- **Reinforcement from researcher feedback**: Researchers rate hypothesis quality; ratings feed back into prompt refinement. Closes the loop between AI generation and human evaluation.
- **Citation verification**: Cross-reference claims in the generated hypothesis against the source abstracts using a separate NIM call. Flags hallucinations before persistence.

---

## 12. Engineering Reflection

### The Core Challenge of Rapid AI Prototyping

The hardest part of building this system was not the LLM integration — it was deciding what *not* to build. Every component has a more sophisticated version: real retrieval instead of mocked data, multi-agent orchestration instead of a sequential pipeline, streaming responses instead of polling, a critic stage instead of single-pass synthesis. Each of these additions is individually justifiable and collectively fatal to a time-bounded project.

The discipline required is to identify the minimum system that clearly demonstrates the architectural ideas, and then build exactly that. The mock data layer is the clearest example: it would have been easy to spend two hours on a scraper that added fragility and distracted from the orchestration pipeline. Instead, the mock is explicitly labelled, architecturally honest, and easy to replace.

### Modular Design as Documentation

One unexpected benefit of the modular folder structure was that it served as implicit documentation during development. When `agent_runner.py` imports from `mock_data.py`, `nemoclaw_client.py`, and `prompts/research_synthesis.py`, the import graph communicates the dependency structure clearly. A new engineer reading the codebase can understand the system's shape from the imports alone, without reading a README.

### Structured Outputs as a Backend Contract

The most technically important decision in this system is the use of `HypothesisOutput` as a shared schema between the NeMoClaw layer and the database layer. In many LLM applications, the model's output is treated as a blob — parsed ad hoc, stored as text, and interpreted by the reader. This approach breaks as soon as the output needs to be queried, compared, or processed programmatically.

Treating the Pydantic schema as the contract between inference and persistence — and building validation, retry logic, and database writes around that contract — is the pattern that separates a demonstration from an engineered system.

### On Honesty in System Design

The clearest articulation of the system's honesty principle is in `MockNeMoClawClient`: it is a named, documented class, not a silent fallback that pretends to be the real thing. It logs a warning. It announces itself in the audit trail. The architectural slot for real NIM inference is present and functional — it just requires an API key to activate. This pattern — being explicit about what is real versus simulated — is important in any system that will be evaluated by someone who might ask "does this actually work?"

---

## 13. Conclusion

Lit-Agent demonstrates that a meaningful autonomous research workflow can be built on a small, coherent backend with clear architectural boundaries. The system is not sophisticated in the sense of being complex — it is sophisticated in the sense of being well-structured, traceable, and extensible.

The key architectural contributions are:

1. **Async task pattern with a visible state machine** — long-running LLM inference is correctly decoupled from synchronous HTTP responses, with state transitions exposed via a polling API.

2. **NeMoClaw as a structured inference layer** — rather than treating the LLM as a text source, the system uses NeMoClaw to enforce a typed output schema, bridging probabilistic inference and deterministic database writes.

3. **Separation of concerns across the pipeline** — retrieval, orchestration, inference, validation, and persistence are cleanly separated into independent modules. Each can be tested, replaced, or extended without touching the others.

4. **Persistence as a first-class concern** — every run is stored, every state transition is logged, and the raw LLM output is retained alongside the parsed fields. The system is auditable by design.

5. **Honest scope management** — the mock retrieval layer and the `MockNeMoClawClient` fallback are explicitly labelled as such. The system is positioned accurately: a lightweight prototype demonstrating the *architecture* of an autonomous research workflow, not a production-grade autonomous scientist.

The architecture is ready to scale: swap SQLite for PostgreSQL, replace `BackgroundTasks` with Celery, replace keyword retrieval with a vector database, and the core orchestration pipeline remains unchanged. That stability under extension is the most reliable signal of a well-designed backend system.

---

*Document version 1.0 — Lit-Agent Backend Prototype*
*Author: Backend Engineering Demo — NVIDIA NeMoClaw + Brev Platform*

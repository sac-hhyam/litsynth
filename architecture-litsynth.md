# LitSynth — Architecture Document

**Research Hypothesis Synthesiser · Backend API Demo**
Stack: FastAPI · SQLite · NVIDIA NIM (LLaMA 3.1 70B) · Python

---

## What It Is

LitSynth is a backend REST API that automates the early ideation phase of academic research. You give it a research topic; it retrieves relevant literature context, calls an LLM to identify the core research gap, and returns a structured, database-persisted hypothesis — all without human intervention after the initial request.

It is testable entirely through the auto-generated Swagger UI at `/docs`. No frontend required.

---

## The Problem It Solves

Identifying a research gap from a set of papers is repetitive, time-consuming work. The steps are structurally identical every time: read abstracts, find what multiple papers fail to address, propose a direction. LitSynth compresses this from hours to seconds by running the synthesis through an LLM and enforcing a typed output schema so the result is immediately machine-readable and comparable across runs.

---

## Current Implementation State

| Capability | Status |
|---|---|
| FastAPI backend with full REST API | **Working** |
| Async pipeline with state machine | **Working** |
| SQLite persistence (tasks, hypotheses, logs) | **Working** |
| LLM inference via NVIDIA NIM API | **Working** (LLaMA 3.1 70B Instruct) |
| Structured output enforcement (Pydantic) | **Working** |
| Offline mock fallback (no API key needed) | **Working** |
| Swagger UI interactive demo | **Working** |
| Brev cloud deployment | **Planned** |
| NeMoClaw native SDK integration | **Planned** |
| Real arXiv / Semantic Scholar retrieval | **Planned** |

Everything runs locally. The LLM calls go out to NVIDIA's hosted NIM API (`integrate.api.nvidia.com`) using a standard API key — no local GPU required.

---

## How It Works

A single research request flows through four stages:

```
POST /api/v1/research/analyze
        │
        │  Returns task_id immediately (HTTP 202)
        │  Pipeline runs in background thread
        ▼
┌─────────────────────────────────────────┐
│           AGENT PIPELINE                │
│                                         │
│  1. PENDING                             │
│     Task created in SQLite              │
│                                         │
│  2. RETRIEVING                          │
│     Topic matched to literature corpus  │
│     3–4 paper abstracts selected        │
│     Context block formatted             │
│                                         │
│  3. SYNTHESIZING                        │
│     Prompt compiled (system + context)  │
│     Sent to NVIDIA NIM API              │
│     LLaMA 3.1 70B generates hypothesis  │
│     JSON validated against schema       │
│     Retried up to 2× on bad output      │
│                                         │
│  4. COMPLETED (or FAILED)               │
│     Hypothesis written to SQLite        │
│     Audit log entries persisted         │
└─────────────────────────────────────────┘
        │
        ▼
GET /api/v1/research/task/{id}/results
  → structured hypothesis JSON
```

The client polls `GET /task/{id}` to watch the state transitions in real time, then fetches the result once the status reaches `COMPLETED`.

---

## API Endpoints

| Method | Endpoint | What It Does |
|---|---|---|
| `GET` | `/health` | Service liveness check |
| `POST` | `/api/v1/research/analyze` | Submit a topic, start the pipeline (202) |
| `GET` | `/api/v1/research/task/{id}` | Poll current status |
| `GET` | `/api/v1/research/task/{id}/results` | Fetch the generated hypothesis |
| `GET` | `/api/v1/research/task/{id}/logs` | Full agent audit trail |
| `GET` | `/api/v1/research/tasks` | List all past tasks |
| `GET` | `/api/v1/research/topics` | List available demo topics |
| `DELETE` | `/api/v1/research/task/{id}` | Remove a task and its output |

### Example Output

```json
POST /api/v1/research/analyze
Body: { "topic": "efficient llm routing strategies" }

→ 202 Accepted
{ "task_id": "abc-123", "status": "PENDING" }

---

GET /api/v1/research/task/abc-123/results

→ 200 OK
{
  "task_id": "abc-123",
  "status": "COMPLETED",
  "hypothesis": {
    "gap_identified": "Existing routing systems make per-token decisions
      that ignore inter-token semantic continuity and cannot adapt to
      distribution shift at inference time.",
    "proposed_architecture": "SeqRouter: a sequence-level routing
      transformer that encodes a sliding window of hidden states via
      a 2-layer cross-attention module, outputting a soft gate vector
      over a pool of draft models updated via EMA accept-rate feedback.",
    "evaluation_metric": "MMLU accuracy degradation vs. routing latency on MT-Bench",
    "confidence_score": "HIGH"
  }
}
```

---

## LLM Integration

LitSynth talks to **NVIDIA NIM** via an OpenAI-compatible `/chat/completions` endpoint. The model is `meta/llama-3.1-70b-instruct`, hosted on NVIDIA's inference infrastructure.

The integration layer (`NeMoClawClient`) does three things beyond a plain API call:

1. **Prompt compilation** — combines a fixed system prompt (persona + strict JSON output rule) with a dynamic user prompt (topic + injected paper abstracts).
2. **Schema enforcement** — the raw LLM response string is parsed and validated against `HypothesisOutput`, a Pydantic model. If the model returns malformed JSON or missing fields, the client retries automatically (up to 2 times).
3. **Fallback** — if no API key is set, a `MockNeMoClawClient` returns realistic deterministic responses. The entire pipeline, state machine, and database layer run identically — only the LLM call is swapped.

The `NeMoClawClient` class is designed as an adapter: swapping the underlying inference backend (e.g., pointing it at a local NIM container on Brev, or integrating the native NeMoClaw SDK) requires only changing the `_call_nim()` method. Nothing else in the system changes.

---

## Structured Output Contract

The LLM is not allowed to return freeform text. The system prompt instructs it to return **only** a JSON object matching this schema:

```python
class HypothesisOutput(BaseModel):
    gap_identified:        str   # shared limitation across surveyed papers
    proposed_architecture: str   # named components, data flow, training paradigm
    evaluation_metric:     str   # "<metric> on <benchmark>"
    confidence_score:      str   # LOW | MEDIUM | HIGH
```

This schema is the contract between the LLM and the database. The same Pydantic model validates the LLM output and drives the columns written to `generated_hypotheses`. The `raw_llm_output` column also stores the original JSON string for audit purposes.

---

## Database

Three tables in SQLite:

**`research_tasks`** — one row per pipeline run. Owns the status state machine. The single source of truth for where a pipeline is.

**`generated_hypotheses`** — one row per completed run (1:1 FK). Only created on success — a missing row is a meaningful signal that the pipeline has not completed.

**`agent_run_logs`** — append-only audit trail. Every state transition writes a row. Accessible via the `/logs` endpoint; useful for debugging and for demonstrating live pipeline activity during a demo.

SQLite was chosen for zero-setup local development. The SQLAlchemy ORM abstracts the database completely — migrating to PostgreSQL means changing `DATABASE_URL` in `.env` and nothing else.

---

## Literature Retrieval (Current)

The retrieval layer is mocked. Four topic corpora are hardcoded in `app/services/mock_data.py`, each containing 3–4 realistic paper abstracts with genuine limitations:

- `efficient llm routing`
- `vision transformer robustness`
- `llm hallucination detection`
- `protein structure prediction`

When a topic is submitted, keyword matching routes it to the closest corpus. The matched abstracts are formatted into a structured context block and injected directly into the LLM prompt.

This is an honest simplification. The architectural slot for real retrieval (arXiv API, Semantic Scholar, PDF parsing) exists and is clearly labelled. Replacing the mock requires swapping one function in `mock_data.py`.

---

## Planned: NeMoClaw on Brev

The intended production path for this project:

1. **Brev instance** — provision a GPU instance on Brev running Ubuntu 22.04.
2. **Local NIM container** — pull and run `nvidia/nim/meta/llama-3.1-70b-instruct` on the Brev instance. Change `NIM_BASE_URL` in `.env` to `http://localhost:8000/v1`.
3. **NeMoClaw SDK** — replace the `_call_nim()` method in `NeMoClawClient` with the native NeMoClaw Python SDK calls, gaining first-class tool binding and guardrails support.
4. **Port forwarding** — `brev port-forward <machine> 8000:8000` exposes the FastAPI server locally for demo access.

The architecture does not change when this transition happens. Only the inference backend changes.

---

## Key Engineering Decisions

**Why async + polling instead of streaming?**
LLM inference takes 5–30 seconds. A synchronous HTTP request held open for that duration is fragile — load balancers time out, clients disconnect. The 202 + polling pattern is standard practice for long-running compute jobs exposed via HTTP. It also makes the internal state transitions externally visible, which is useful for a demo.

**Why SQLite instead of PostgreSQL?**
Zero infrastructure setup. For a single-developer demo with sequential pipeline runs, SQLite's write serialisation is not a constraint. The ORM abstraction means the upgrade path is one line in `.env`.

**Why a mock retrieval layer instead of real arXiv?**
PDF scraping introduces network failures, encoding edge cases, and rate limits — all of which would consume more time to handle gracefully than the retrieval adds in demo value. The mock produces the same quality of LLM input as a real retrieval system would. The architectural separation makes swapping it straightforward when the project extends beyond MVP.

**Why no frontend?**
FastAPI's Swagger UI (`/docs`) is a fully functional interactive interface for a backend demo. Building a React frontend shifts focus from backend engineering to UI work without improving the demo's ability to communicate the system's architecture.

---

## Running It

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add: NVIDIA_API_KEY=nvapi-...

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Open: http://localhost:8000/docs
```

No API key? Leave it blank. The mock client activates automatically and the full pipeline still runs.

---

## Project Structure

```
backend/
├── main.py                          # App entry point, lifespan DB init
├── app/
│   ├── api/routes.py                # All REST endpoints
│   ├── core/config.py               # Settings from .env
│   ├── db/
│   │   ├── database.py              # SQLAlchemy engine + session factory
│   │   └── models.py                # ORM models (3 tables)
│   ├── prompts/research_synthesis.py # System + user prompt templates
│   ├── schemas/pydantic.py          # Request/response + HypothesisOutput
│   └── services/
│       ├── agent_runner.py          # Pipeline state machine
│       ├── mock_data.py             # Literature corpus + keyword router
│       └── nemoclaw_client.py       # NIM adapter + mock fallback
└── requirements.txt
```

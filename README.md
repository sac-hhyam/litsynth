# Lit-Agent: Autonomous Research Hypothesis Generator

A backend-only AI research assistant powered by **NVIDIA NeMoClaw** and deployed on **Brev**.

Submit a research topic → the agent retrieves literature context → invokes NeMoClaw to synthesise a research gap → persists a structured experiment hypothesis to SQLite.

---

## Architecture

```
Client (curl / Swagger UI)
        │
        ▼
  FastAPI Router  ──── POST /analyze ──► BackgroundTask
        │                                      │
        │                                      ▼
        │                              AgentRunner (state machine)
        │                              PENDING → RETRIEVING → SYNTHESIZING → COMPLETED
        │                                      │
        │                            ┌─────────┴──────────┐
        │                            ▼                    ▼
        │                    mock_data.py          NeMoClawClient
        │                  (literature corpus)    ┌───────────────┐
        │                                         │  NVIDIA NIM   │
        │                                         │  /chat/       │
        │                                         │  completions  │
        │                                         └───────────────┘
        │                                                │
        │                                       Pydantic validation
        │                                       (HypothesisOutput)
        │                                                │
        │                                               ▼
        └──────── GET /task/{id}/results ◄──── SQLite (SQLAlchemy)
```

**NeMoClaw is responsible for:**
- Prompt compilation (system + user template fusion)
- Inference against the NIM endpoint (OpenAI-compatible API)
- Structured-output enforcement via Pydantic schema validation
- Retry logic on malformed JSON responses

---

## Quick Start (Local)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # add your NVIDIA_API_KEY
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** for the Swagger UI.

> **No API key?** The system automatically switches to `MockNeMoClawClient`,
> which returns realistic deterministic hypotheses. The full pipeline, state
> machine, and persistence layer remain identical.

---

## Demo Flow

```bash
# 1. Check available demo topics
curl http://localhost:8000/api/v1/research/topics

# 2. Submit a research topic (returns task_id immediately — 202 Accepted)
curl -X POST http://localhost:8000/api/v1/research/analyze \
  -H "Content-Type: application/json" \
  -d '{"topic": "efficient llm routing strategies for multi-task inference"}'

# 3. Poll the state machine
curl http://localhost:8000/api/v1/research/task/<task_id>

# 4. Fetch the structured hypothesis
curl http://localhost:8000/api/v1/research/task/<task_id>/results

# 5. Review the agent audit trail
curl http://localhost:8000/api/v1/research/task/<task_id>/logs
```

---

## Brev Deployment

```bash
# On your Brev instance
git clone <repo> && cd demo/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in NVIDIA_API_KEY

uvicorn main:app --host 0.0.0.0 --port 8000

# From your LOCAL machine — port-forward to access Swagger UI in browser
brev port-forward <machine-name> 8000:8000
# Open: http://localhost:8000/docs
```

For a local NIM container on Brev, set `NIM_BASE_URL=http://localhost:8000/v1` in `.env`.

---

## API Reference

| Method   | Endpoint                              | Description                           |
|----------|---------------------------------------|---------------------------------------|
| `GET`    | `/health`                             | Service health check                  |
| `POST`   | `/api/v1/research/analyze`            | Submit topic, start pipeline (202)    |
| `GET`    | `/api/v1/research/task/{id}`          | Poll pipeline status                  |
| `GET`    | `/api/v1/research/task/{id}/results`  | Fetch structured hypothesis output    |
| `GET`    | `/api/v1/research/task/{id}/logs`     | Agent run audit trail                 |
| `GET`    | `/api/v1/research/tasks`             | List all tasks                        |
| `GET`    | `/api/v1/research/topics`            | List available demo topics            |
| `DELETE` | `/api/v1/research/task/{id}`          | Delete task and its output            |

---

## Folder Structure

```
backend/
├── app/
│   ├── api/routes.py           # FastAPI endpoints + BackgroundTasks
│   ├── core/config.py          # Settings from .env
│   ├── db/
│   │   ├── database.py         # SQLAlchemy engine & session factory
│   │   └── models.py           # ResearchTask, GeneratedHypothesis, AgentRunLog
│   ├── prompts/
│   │   └── research_synthesis.py  # System & user prompt templates
│   ├── schemas/pydantic.py     # Request/response + NeMoClaw output contracts
│   ├── services/
│   │   ├── agent_runner.py     # Pipeline state machine
│   │   ├── mock_data.py        # Literature corpus (4 topics, 3–4 papers each)
│   │   └── nemoclaw_client.py  # NeMoClaw / NIM adapter + mock fallback
│   └── utils/logging.py        # Structured logging config
├── main.py                     # App entry point
├── requirements.txt
└── .env.example
```

---

## Stack

| Component    | Technology                         | Why                                          |
|--------------|------------------------------------|----------------------------------------------|
| API          | FastAPI + uvicorn                  | Async, auto-Swagger, production-grade        |
| Persistence  | SQLite + SQLAlchemy 2.0            | Zero setup, relational, inspectable          |
| AI Runtime   | NVIDIA NeMoClaw (NIM API)          | Structured output enforcement, NIM inference |
| Async        | FastAPI `BackgroundTasks`          | Decouples HTTP from LLM latency              |
| Validation   | Pydantic v2                        | Strict schema enforcement on agent output    |

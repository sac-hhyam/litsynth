# LitSynth

**Autonomous Research Hypothesis Generator**

Submit any research topic → LitSynth fetches live papers from arXiv → identifies the core research gap → returns a structured, database-persisted experiment hypothesis via REST API.

Built with FastAPI, SQLite, and NVIDIA NIM (LLaMA 3.1 70B). Designed for NeMoClaw agent orchestration on Brev.

---

## How It Works

```
POST /api/v1/research/analyze
        │
        │  202 Accepted  (immediate)
        ▼
  BackgroundTask
        │
        ▼
  AgentRunner  ── PENDING → RETRIEVING → SYNTHESIZING → COMPLETED
        │                       │                │
        │                       ▼                ▼
        │               arXiv API         NeMoClawClient
        │               (live papers)     → NVIDIA NIM
        │                                 → Pydantic validation
        │                                 → HypothesisOutput schema
        ▼
  SQLite  (ResearchTask + GeneratedHypothesis + AgentRunLog)
        │
        ▼
  GET /api/v1/research/task/{id}/results
```

Every request is async. The HTTP response returns a `task_id` instantly. You poll the status endpoint to watch the pipeline transition through states in real time, then fetch the structured result once `COMPLETED`.

---

## Quickstart

**Requirements:** Python 3.10+, an [NVIDIA API key](https://build.nvidia.com) (free tier)

```bash
git clone https://github.com/YOUR_USERNAME/litsynth.git
cd litsynth/backend

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your NVIDIA_API_KEY to .env

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** — Swagger UI is the interactive interface.

> **No API key?** The system falls back to `MockNeMoClawClient` automatically.  
> The full pipeline, state machine, arXiv retrieval, and persistence all run identically.

---

## Demo

```bash
# 1. Submit a topic — any topic, arXiv handles retrieval
curl -X POST http://localhost:8000/api/v1/research/analyze \
  -H "Content-Type: application/json" \
  -d '{"topic": "swarm optimization for IoT network security"}'

# → {"task_id": "abc-123", "status": "PENDING", ...}

# 2. Poll the state machine
curl http://localhost:8000/api/v1/research/task/abc-123

# → {"status": "SYNTHESIZING", ...}

# 3. Fetch the hypothesis once COMPLETED
curl http://localhost:8000/api/v1/research/task/abc-123/results

# 4. Inspect the full agent audit trail
curl http://localhost:8000/api/v1/research/task/abc-123/logs
```

**Example output:**
```json
{
  "hypothesis": {
    "gap_identified": "Current swarm-based IDS systems optimise detection rate in isolation, ignoring energy constraints of resource-limited IoT edge nodes under adversarial conditions.",
    "proposed_architecture": "EnergyAware-PSO-IDS: a multi-objective particle swarm optimiser with an adaptive inertia weight controller that jointly minimises false-negative rate and node energy consumption...",
    "evaluation_metric": "F1 detection rate vs. energy overhead on N-BaIoT benchmark",
    "confidence_score": "HIGH"
  }
}
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/research/analyze` | Submit topic, start pipeline (202) |
| `GET` | `/api/v1/research/task/{id}` | Poll pipeline status |
| `GET` | `/api/v1/research/task/{id}/results` | Fetch structured hypothesis |
| `GET` | `/api/v1/research/task/{id}/logs` | Agent run audit trail |
| `GET` | `/api/v1/research/tasks` | List all tasks |
| `GET` | `/api/v1/research/topics` | List hardcoded demo topics |
| `DELETE` | `/api/v1/research/task/{id}` | Delete task and output |

---

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   └── routes.py               # All endpoints + BackgroundTask dispatch
│   ├── core/
│   │   └── config.py               # Settings loaded from .env
│   ├── db/
│   │   ├── database.py             # SQLAlchemy engine, session factory
│   │   └── models.py               # ResearchTask, GeneratedHypothesis, AgentRunLog
│   ├── prompts/
│   │   └── research_synthesis.py   # System + user prompt templates
│   ├── schemas/
│   │   └── pydantic.py             # Request/response + HypothesisOutput contract
│   ├── services/
│   │   ├── agent_runner.py         # Pipeline state machine
│   │   ├── arxiv_client.py         # Live arXiv paper retrieval
│   │   ├── mock_data.py            # Fallback hardcoded corpus + unified get_papers()
│   │   └── nemoclaw_client.py      # NIM adapter, schema enforcement, retry logic
│   └── utils/
│       └── logging.py              # Structured logging config
├── main.py                         # App entry point
├── requirements.txt
└── .env.example
```

---

## Stack

| Layer | Technology | Role |
|---|---|---|
| API | FastAPI + uvicorn | Async HTTP, auto-Swagger, BackgroundTasks |
| AI Inference | NVIDIA NIM — LLaMA 3.1 70B | LLM inference via OpenAI-compatible API |
| Orchestration | NeMoClawClient adapter | Prompt compilation, schema enforcement, retries |
| Retrieval | arXiv API | Live paper fetching for any topic |
| Persistence | SQLite + SQLAlchemy 2.0 | Task state, hypothesis output, audit logs |
| Validation | Pydantic v2 | Strict structured output contract |

---

## Brev Deployment

```bash
# SSH into your Brev instance
brev shell <workspace-name>

git clone https://github.com/YOUR_USERNAME/litsynth.git
cd litsynth/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in NVIDIA_API_KEY

uvicorn main:app --host 0.0.0.0 --port 8000

# From your local machine — access Swagger UI in browser
brev port-forward <workspace-name> --port 8000
# → http://localhost:8000/docs
```

To use a local NIM container instead of the hosted API, set `NIM_BASE_URL=http://localhost:8000/v1` in `.env`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | — | From [build.nvidia.com](https://build.nvidia.com) |
| `NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NIM endpoint (swap for local on Brev) |
| `NIM_MODEL` | `meta/llama-3.1-70b-instruct` | Model served by NIM |
| `DATABASE_URL` | `sqlite:///./litsynth.db` | SQLAlchemy connection string |
| `MAX_SYNTHESIS_TOKENS` | `1024` | Max tokens for LLM response |
| `LLM_TEMPERATURE` | `0.4` | Inference temperature |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

# LitSynth

**Autonomous AI Research Hypothesis Generator**
Powered by NVIDIA NeMoClaw · OpenShell Privacy Sandbox · Discord Interface

---

## Overview

LitSynth is a fully autonomous research hypothesis engine. A researcher types `!synthesize <topic>` in Discord; within seconds the system retrieves live academic abstracts from OpenAlex, passes them through an NVIDIA NeMoClaw / OpenShell sandbox, and returns a structured hypothesis — research gap, proposed architecture, evaluation metric, and confidence score — either as a Discord message or a downloadable Markdown attachment.

The system is built on NVIDIA NeMoClaw v0.0.46 with the OpenClaw agent runtime inside a kernel-level OpenShell sandbox. All NIM inference flows through an OpenShell Privacy Router at `inference.local`, which injects NVIDIA credentials and forwards to `integrate.api.nvidia.com`. Paper retrieval runs on the Brev host (unrestricted network); synthesis runs inside the sandboxed environment. NeMo Guardrails (Colang flows) provide off-topic detection before each synthesis call.

---

## Architecture

```
User: !synthesize <topic>
          │
          ▼
  ┌───────────────────┐
  │   Discord Server  │
  └────────┬──────────┘
           │
           ▼
  ┌────────────────────────────────────────────────┐
  │  bot.py  (Brev HOST — unrestricted network)    │
  │                                                │
  │  1. Calls OpenAlex API                         │
  │     → fetches 4 paper abstracts                │
  │  2. Serialises papers to JSON                  │
  └────────────────┬───────────────────────────────┘
                   │  papers_json via --context flag
                   │
                   ▼
  nemoclaw litsynth-sandbox exec --no-tty \
    --workdir /sandbox/.openclaw/skills/litsynth \
    -- python3 synthesise.py "<topic>" --context '<papers_json>'
                   │
                   ▼
  ┌────────────────────────────────────────────────┐
  │  OpenShell Sandbox (litsynth-sandbox)          │
  │  Network enforced via proxy 10.200.0.1:3128    │
  │                                                │
  │  synthesise.py                                 │
  │  ├─ Formats context block from pre-fetched     │
  │  │  papers                                     │
  │  ├─ Phase A: lightweight NeMo Guardrails probe │
  │  │  (off-topic detection via LLMRails)         │
  │  └─ Phase B: raw NIM call via inference.local  │
  │     ↓                                          │
  │  OpenShell Privacy Router                      │
  │  inference.local:443 → integrate.api.nvidia.com│
  │  (credentials injected by router)              │
  └────────────────┬───────────────────────────────┘
                   │  JSON on stdout
                   ▼
  ┌────────────────────────────────────────────────┐
  │  bot.py parses stdout JSON                     │
  │  ≤2000 chars → Discord message                 │
  │  >2000 chars → hypothesis_<topic>.md attachment│
  └────────────────────────────────────────────────┘
```

**Deployed on:** Brev GPU instance — GCP `g2-standard-4`, NVIDIA L4 24 GB, Toronto region

---

## Quick Start

### Prerequisites

- Brev account with a `g2-standard-4` (NVIDIA L4) instance running
- SSH access via `~/.brev/brev.pem`
- NVIDIA NIM API key (`nvapi-...`)
- Discord bot token

### 1. Clone and install

```bash
git clone <repo> && cd demo/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in: NVIDIA_API_KEY, DISCORD_BOT_TOKEN, NEMOCLAW_SANDBOX
```

### 2. Install NeMoClaw skill

```bash
nemoclaw litsynth-sandbox skill install skills/litsynth
```

### 3. Verify sandbox connectivity

```bash
nemoclaw litsynth-sandbox doctor
nemoclaw litsynth-sandbox exec --no-tty --workdir /sandbox/.openclaw/skills/litsynth \
  -- python3 synthesise.py "efficient llm routing"
```

### 4. Run the Discord bot

```bash
python bot.py
```

The bot will come online in your Discord server. Type `!synthesize <topic>` in any channel the bot has access to.

### 5. (Optional) Verify host-side paper fetch

```bash
python -c "
import asyncio, httpx
async def test():
    async with httpx.AsyncClient() as c:
        r = await c.get('https://api.openalex.org/works?filter=title.search:llm routing&per_page=4')
        print(r.status_code, len(r.json()['results']), 'papers')
asyncio.run(test())
"
```

---

## NeMoClaw Commands Reference

| Category | Command | Description |
|---|---|---|
| **Sandbox** | `nemoclaw list` | List all sandboxes |
| | `nemoclaw status` | Global NeMoClaw status |
| | `nemoclaw litsynth-sandbox doctor` | Health-check the sandbox |
| | `nemoclaw litsynth-sandbox recover` | Attempt auto-recovery |
| | `nemoclaw litsynth-sandbox rebuild` | Full sandbox rebuild |
| | `nemoclaw litsynth-sandbox destroy --yes` | Destroy sandbox |
| **Skill** | `nemoclaw litsynth-sandbox skill install skills/litsynth` | Install synthesis skill |
| **Policy** | `nemoclaw litsynth-sandbox policy-add --from-file config/openshell_policy.yaml --dry-run` | Dry-run policy change |
| | `nemoclaw litsynth-sandbox policy-add --from-file config/openshell_policy.yaml` | Apply egress policy |
| | `nemoclaw litsynth-sandbox policy-list` | List active policies |
| **Channels** | `nemoclaw litsynth-sandbox channels add discord` | Add Discord channel |
| | `nemoclaw litsynth-sandbox channels list` | List registered channels |
| **Inference** | `nemoclaw inference get --json` | Inspect inference config |
| **Exec** | `nemoclaw litsynth-sandbox exec --no-tty --workdir /sandbox/.openclaw/skills/litsynth -- python3 synthesise.py "<topic>"` | Run synthesis directly |
| **Cleanup** | `docker stop nemoclaw-openshell-gateway && docker rm nemoclaw-openshell-gateway` | Remove gateway container |

---

## Output Schema

```json
{
  "gap_identified": "Specific shared limitation identified across papers",
  "proposed_architecture": "Named architecture with components and training paradigm",
  "evaluation_metric": "Metric on benchmark (e.g. MMLU accuracy at 10ms latency)",
  "confidence_score": "HIGH | MEDIUM | LOW",
  "topic": "The submitted research topic",
  "papers_used": 4,
  "source": "host-fetched | openalex | semantic_scholar"
}
```

Discord output behaviour:
- Result ≤ 2000 characters → posted as a Discord message
- Result > 2000 characters → uploaded as `hypothesis_<topic>.md`

---

## Environment Variables

| Variable | Description | Example |
|---|---|---|
| `NVIDIA_API_KEY` | NVIDIA NIM API key | `nvapi-...` |
| `NIM_BASE_URL` | NIM inference endpoint | `https://inference.local/v1` |
| `NIM_MODEL` | Model identifier | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` |
| `DISCORD_BOT_TOKEN` | Discord bot token | `MTI3...` |
| `NEMOCLAW_SANDBOX` | Sandbox name | `litsynth-sandbox` |
| `SYNTHESIS_TIMEOUT` | Max seconds per synthesis call | `120` |
| `DATABASE_URL` | SQLite path (legacy) | `sqlite:///./litsynth.db` |
| `MAX_SYNTHESIS_TOKENS` | Token limit for NIM response | `1024` |
| `LLM_TEMPERATURE` | Sampling temperature | `0.4` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

> **Note on `NIM_BASE_URL`:** Inside the sandbox, always use `https://inference.local/v1`. The OpenShell Privacy Router intercepts this hostname, injects NVIDIA credentials, and forwards to `integrate.api.nvidia.com`. Do not call `integrate.api.nvidia.com` directly from inside the sandbox — it is blocked by the proxy.

---

## Repository Structure

```
backend/
├── main.py                           # FastAPI app (decommissioned on this branch)
├── bot.py                            # Discord bot — primary interface
├── skills/
│   └── litsynth/
│       ├── SKILL.md                  # NeMoClaw skill manifest (YAML frontmatter)
│       ├── synthesise.py             # Synthesis skill (runs inside sandbox)
│       └── requirements.txt
├── config/
│   ├── openshell_policy.yaml         # Network egress policy (litsynth-policy)
│   ├── rails.yaml                    # NeMo Guardrails config
│   └── synthesiser.co                # Colang flow definitions
├── app/
│   ├── api/routes.py
│   ├── core/config.py
│   ├── db/                           # SQLAlchemy models + migrations
│   ├── prompts/research_synthesis.py
│   ├── schemas/pydantic.py
│   └── services/
│       ├── nemoclaw_client.py        # NeMoGuardrailsClient (two-phase guard + NIM)
│       ├── mock_data.py              # Mock corpus fallback
│       └── semantic_scholar_client.py
└── .env.example
```

---

## Branch Structure

| Branch | Interface | Key Differences |
|---|---|---|
| `main` | FastAPI REST API | arXiv retrieval (replaced), Swagger UI at `/docs`, synchronous polling pattern |
| `feat/nemoclaw-discord-agent` | Discord bot | OpenAlex on host, NeMoClaw sandbox exec, OpenShell Privacy Router, NeMo Guardrails two-phase pipeline |

The `main` branch is an earlier prototype with a decommissioned FastAPI interface. All active development is on `feat/nemoclaw-discord-agent`.

---

## Stack

| Component | Technology |
|---|---|
| Sandbox orchestration | NVIDIA NeMoClaw v0.0.46 |
| Agent runtime | OpenClaw v2026.4.24 |
| Kernel sandbox | OpenShell |
| LLM inference | NVIDIA NIM — `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` |
| Paper retrieval | OpenAlex API (free, no key required) |
| User interface | Discord (`discord.py`) |
| Guardrails | NeMo Guardrails (Colang) |
| Task persistence | SQLite + SQLAlchemy |
| HTTP client | httpx, Pydantic |
| Language | Python 3.10 |
| Deployment | Brev — GCP `g2-standard-4`, NVIDIA L4 24 GB, Toronto |

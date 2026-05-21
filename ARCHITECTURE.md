# LitSynth — System Architecture

**Autonomous AI Research Hypothesis Generator**
NVIDIA NeMoClaw + OpenShell + Discord

---

## 1. System Overview

LitSynth is a three-layer system: a Discord bot on the Brev host, a NeMoClaw-managed OpenShell sandbox, and an NVIDIA NIM inference endpoint reached through the OpenShell Privacy Router. These three layers are intentionally decoupled — each has distinct network access, distinct responsibilities, and distinct failure modes.

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Discord Host (Brev g2-standard-4, unrestricted net) │
│                                                                 │
│  bot.py                                                         │
│  ├─ Listens for !synthesize commands                            │
│  ├─ Fetches papers from api.openalex.org                        │
│  └─ Shells out to NeMoClaw exec → parses stdout JSON            │
└─────────────────────────┬───────────────────────────────────────┘
                          │  nemoclaw litsynth-sandbox exec
                          │  (papers JSON via --context flag)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 — OpenShell Sandbox (litsynth-sandbox)                 │
│                                                                 │
│  synthesise.py                                                  │
│  ├─ Parses pre-fetched papers from --context argument           │
│  ├─ Phase A: NeMo Guardrails probe (off-topic detection)        │
│  └─ Phase B: Direct NIM call via inference.local                │
│                                                                 │
│  Network policy: all egress via proxy 10.200.0.1:3128           │
│  Only inference.local:443 is reachable                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │  HTTPS CONNECT to inference.local:443
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3 — OpenShell Privacy Router                             │
│                                                                 │
│  Intercepts inference.local:443                                 │
│  Injects NVIDIA_API_KEY credentials                             │
│  Forwards to integrate.api.nvidia.com                           │
│                                                                 │
│  Model: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. End-to-End Data Flow

### Step-by-step: `!synthesize efficient llm routing`

```
1. USER types in Discord:
   !synthesize efficient llm routing

2. bot.py (host) receives Discord message event

3. bot.py → GET https://api.openalex.org/works
             ?filter=title.search:efficient llm routing
             &per_page=4
   Response: [{title, authors, abstract}, × 4]
   papers_json = json.dumps([...])

4. bot.py spawns subprocess:
   nemoclaw litsynth-sandbox exec \
     --no-tty \
     --workdir /sandbox/.openclaw/skills/litsynth \
     -- python3 synthesise.py "efficient llm routing" \
        --context '<papers_json>'

5. NeMoClaw resolves sandbox: litsynth-sandbox
   Enters OpenShell container environment

6. synthesise.py starts inside sandbox:
   a. Parses --context argument → list of paper dicts
   b. Formats context block:
      "Paper 1: <title>\nAbstract: <abstract>\n\n..."

7. Phase A — NeMo Guardrails off-topic probe:
   LLMRails.generate(
     messages=[{"role": "user", "content": topic}]
   )
   Colang flows evaluate:
     define user ask off topic → bot refuse
     define user synthesise research gap → bot respond
   If off-topic: synthesise.py exits with error JSON

8. Phase B — Direct NIM synthesis call:
   POST https://inference.local/v1/chat/completions
   Headers:
     Authorization: Bearer openshell-managed
     Content-Type: application/json
   Body:
     model: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
     messages:
       - {role: system, content: SYNTHESIS_SYSTEM_PROMPT}
       - {role: user, content: formatted context + topic}
     max_tokens: 1024
     temperature: 0.4

9. OpenShell Privacy Router intercepts inference.local:443
   - Strips "Authorization: Bearer openshell-managed"
   - Injects "Authorization: Bearer nvapi-..."
   - Forwards to https://integrate.api.nvidia.com/v1/chat/completions

10. NIM responds with JSON string in assistant message

11. synthesise.py parses response:
    - Slices from first '{' to last '}'
    - json.loads() → validates required fields
    - Prints JSON to stdout

12. bot.py captures stdout, parses JSON

13. Result dispatch:
    len(result_str) ≤ 2000:
      await channel.send(formatted_result)
    len(result_str) > 2000:
      await channel.send(file=discord.File(
        BytesIO(result_str.encode()),
        filename="hypothesis_efficient_llm_routing.md"
      ))
```

---

## 3. OpenShell Sandbox Internals

### What OpenShell is

OpenShell is a kernel-level sandbox environment managed by NeMoClaw. It provides process isolation, filesystem namespacing, and — critically — network egress control enforced by a proxy at `10.200.0.1:3128`. All outbound TCP connections from within the sandbox are intercepted by this proxy.

### Network access model

```
INSIDE SANDBOX                    PROXY              EXTERNAL
────────────────                  ─────              ────────

inference.local:443     ──CONNECT──►  10.200.0.1:3128
                                         │
                           200 Connection Established
                                         │
                                         └──► integrate.api.nvidia.com
                                              (credentials injected)

api.openalex.org        ──CONNECT──►  10.200.0.1:3128
                                         │
                                       403 Forbidden
                                       (not in active policy)
```

The proxy enforces allow-list semantics: only endpoints granted by the active policy preset are reachable. In NeMoClaw v0.0.46, only built-in presets (e.g., `discord`) propagate fully to the gateway proxy. Custom `--from-file` policies are recorded locally but do not become active on the proxy — which is why `api.openalex.org` is fetched on the host instead of inside the sandbox.

### Skill installation path

```
Host filesystem:    backend/skills/litsynth/
                    ├── SKILL.md          ← NeMoClaw manifest
                    ├── synthesise.py
                    └── requirements.txt

After install:      /sandbox/.openclaw/skills/litsynth/
                    ├── SKILL.md
                    ├── synthesise.py
                    └── requirements.txt
```

The `SKILL.md` manifest requires YAML frontmatter:

```yaml
---
name: litsynth
version: 0.1.0
description: Research gap synthesis skill
entrypoint: synthesise.py
dependencies:
  - httpx
  - pydantic
env:
  - NIM_BASE_URL
  - SYNTHESIS_TIMEOUT
---
```

---

## 4. OpenShell Privacy Router

The Privacy Router is OpenShell's mechanism for injecting managed credentials into sandbox egress traffic without exposing secrets inside the sandbox.

```
synthesise.py sets:
  NIM_BASE_URL = "https://inference.local/v1"
  api_key      = "openshell-managed"           ← placeholder

                        │
                        ▼ HTTPS CONNECT
              proxy at 10.200.0.1:3128
                        │
               hostname = inference.local
                        │
                 matches Privacy Router rule
                        │
                        ▼
        Router strips: Authorization: Bearer openshell-managed
        Router injects: Authorization: Bearer nvapi-<real-key>
                        │
                        ▼
        https://integrate.api.nvidia.com/v1/chat/completions
```

The real `NVIDIA_API_KEY` is stored in the NeMoClaw credential store on the host — never written inside the sandbox filesystem. The sandbox only ever sees the `openshell-managed` placeholder. This is the privacy guarantee: compromising the sandbox does not expose NVIDIA credentials.

---

## 5. NeMo Guardrails Two-Phase Pipeline

### Why two phases

A naive single-phase approach (calling LLMRails for both guardrail checking and synthesis output generation) fails because:

- NeMo Guardrails Phase 3 (`generate_bot_message`) rewrites the assistant response into natural language prose, destroying the JSON structure needed for machine parsing.
- With ~100 preamble tokens consumed by the Colang runtime, the 1024-token limit is reached mid-JSON, producing `Unterminated string` parse errors.

The solution is to separate the concerns into two independent calls:

```
Phase A — Guardrail probe (LLMRails)
─────────────────────────────────────
Input:  user message (topic string only)
Call:   LLMRails.generate(messages=[{"role":"user","content":topic}])

Colang flows evaluated:
  define user ask off topic
    example: "tell me a joke"
    example: "write me a poem"
    → bot refuse

  define user synthesise research gap
    example: "efficient llm routing"
    example: "transformer attention optimisation"
    → bot respond

Output: intent classification — pass or refuse
        If refused: exit with {"error": "off-topic", ...}

Phase B — Direct NIM synthesis (httpx)
───────────────────────────────────────
Input:  formatted context block (pre-fetched papers + topic)
Call:   POST https://inference.local/v1/chat/completions
        (bypasses LLMRails Phase 3 entirely)

System prompt: SYNTHESIS_SYSTEM_PROMPT
  "You are a research synthesis engine.
   Return ONLY valid JSON matching this schema: {...}
   Do not include any other text."

User prompt: context block + topic

Output: raw assistant message → JSON parse → HypothesisOutput
```

### Colang flow structure (`config/synthesiser.co`)

```colang
define user ask off topic
  "tell me a joke"
  "write me a poem"
  "what's the weather"

define flow handle off topic
  user ask off topic
  bot refuse

define user synthesise research gap
  "efficient llm routing"
  "transformer attention optimisation"
  "multimodal fusion architectures"

define flow synthesise and propose flow
  user synthesise research gap
  bot respond
```

### rails.yaml configuration

```yaml
models:
  - type: main
    engine: nim
    model: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning

rails:
  input:
    flows:
      - handle off topic
  # NOTE: research synthesis guardrail is NOT in output.flows
  # Output rails invoke Phase 3 (generate_bot_message) which
  # rewrites JSON to prose. Only input rails are used.
```

---

## 6. File Responsibilities

| File | Layer | Responsibility |
|---|---|---|
| `bot.py` | Host | Discord event loop, OpenAlex paper fetch, NeMoClaw exec subprocess, stdout parsing, Discord message dispatch |
| `skills/litsynth/synthesise.py` | Sandbox | Context formatting, NeMo Guardrails Phase A probe, NIM Phase B synthesis call, JSON output |
| `skills/litsynth/SKILL.md` | Sandbox | NeMoClaw skill manifest — name, version, entrypoint, deps, env vars |
| `config/openshell_policy.yaml` | Config | Network egress allow-list (recorded locally; not gateway-active in v0.0.46) |
| `config/rails.yaml` | Config | NeMo Guardrails model config and input rail flow registrations |
| `config/synthesiser.co` | Config | Colang intent definitions and flow mappings |
| `app/services/nemoclaw_client.py` | Legacy | NeMoGuardrailsClient used in FastAPI branch; two-phase guard + NIM |
| `app/prompts/research_synthesis.py` | Legacy | SYNTHESIS_SYSTEM_PROMPT and user template strings |
| `app/db/` | Legacy | SQLAlchemy models — ResearchTask, GeneratedHypothesis, AgentRunLog |
| `main.py` | Legacy | FastAPI application (decommissioned on this branch) |

---

## 7. Network Policy Configuration

The egress policy is defined in `config/openshell_policy.yaml`:

```yaml
preset:
  name: litsynth-policy          # RFC 1123 — lowercase, hyphens only
  endpoints:
    - host: inference.local      # Privacy Router (NIM access)
    - host: api.openalex.org     # Paper retrieval (host-fetched; listed for future)
    - host: discord.com          # Discord API
    - host: gateway.discord.gg   # Discord gateway
  binaries:
    - python3
    - pip3
```

Key constraints discovered in NeMoClaw v0.0.46:
- `network_policies.egress.default: deny` is not a valid field — deny-by-default is implicit.
- Endpoint entries must be structs (`- host: <hostname>`), not plain strings.
- `preset.name` is required and must match RFC 1123 label format.
- Custom `--from-file` policies are recorded locally but do not propagate to the OpenShell gateway proxy. Only built-in presets (e.g., `discord`) are applied at the proxy level.

---

## 8. Host-Sandbox Split: Why Papers Are Fetched on the Host

The initial design fetched papers inside the sandbox via the OpenAlex API. This was blocked by the proxy (`403 Forbidden` on `api.openalex.org`) because the custom `litsynth-policy` was not gateway-active.

Rather than waiting for policy propagation (which proved unreliable in v0.0.46), the architecture was split:

```
BEFORE (broken):                      AFTER (current):

Sandbox:                              Host:
  synthesise.py                         bot.py
    → GET api.openalex.org                → GET api.openalex.org ✓
       ❌ 403 from proxy                  → papers_json = [...]
                                          → nemoclaw exec \
                                              --context papers_json

                                      Sandbox:
                                        synthesise.py
                                          → parse --context arg ✓
                                          → call inference.local ✓
```

This split uses the two environments according to their actual capabilities:
- Host: unrestricted network, paper retrieval
- Sandbox: managed NIM access via Privacy Router, synthesis

---

## 9. Inference Configuration

| Parameter | Value |
|---|---|
| Endpoint (sandbox) | `https://inference.local/v1/chat/completions` |
| Endpoint (external) | `https://integrate.api.nvidia.com/v1/chat/completions` |
| Model | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` |
| Auth in sandbox | `Bearer openshell-managed` (Privacy Router rewrites) |
| Max tokens | 1024 |
| Temperature | 0.4 |
| Papers per synthesis | 4 |

---

## 10. Persistence Layer (Retained from FastAPI Branch)

SQLite + SQLAlchemy models remain in the codebase for potential re-integration. They are not used by the Discord bot path.

```
research_tasks
  id (UUID PK) | topic | status | error_message | created_at | updated_at

generated_hypotheses
  id | task_id (FK) | gap_identified | proposed_architecture
     | evaluation_metric | confidence_score | raw_llm_output

agent_run_logs
  id | task_id (FK) | stage | message | timestamp
```

The `raw_llm_output` field stores the unmodified assistant message alongside the parsed fields, enabling post-hoc analysis of model behaviour across runs.

---

## 11. Deployment Topology

```
GCP Toronto Region
└── Brev GPU Instance: g2-standard-4
    ├── NVIDIA L4 24GB (not used for inference — NIM is hosted)
    ├── Ubuntu 22.04
    │
    ├── Host processes:
    │   ├── bot.py (Python 3.10, discord.py)
    │   └── nemoclaw daemon
    │
    ├── Docker containers:
    │   ├── nemoclaw-openshell-gateway  ← proxy at 10.200.0.1:3128
    │   └── litsynth-sandbox container  ← OpenShell + OpenClaw runtime
    │
    └── Credentials:
        ├── NVIDIA_API_KEY (host .env, injected by Privacy Router)
        └── DISCORD_BOT_TOKEN (host .env, used by bot.py)

External services:
├── api.openalex.org       ← paper retrieval (host only)
└── integrate.api.nvidia.com ← NIM inference (via Privacy Router)
```

SSH access: `ssh -i ~/.brev/brev.pem ubuntu@<instance-ip>` (used when `brev shell` is unavailable due to control plane TLS timeouts).

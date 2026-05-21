# LitSynth — Development Log

Chronological record of bugs encountered, root causes, fixes applied, and architectural decisions made during the build of LitSynth.

---

## Phase 1: Initial FastAPI + NeMo Guardrails Setup

The first phase built the foundation: a FastAPI backend, NeMo Guardrails integration, and live paper retrieval. Four separate bugs were hit before the pipeline was stable.

---

### Bug 1 — NeMo Guardrails 401 Unauthorized

**What failed:**
All calls to the NIM endpoint through `LLMRails` returned HTTP 401. The NVIDIA API key was present in the environment but not being used.

**Why it failed:**
NeMo Guardrails initialises its underlying LLM client using LangChain's `ChatOpenAI`. By default, `ChatOpenAI` reads from the `OPENAI_API_KEY` environment variable. The project set `NVIDIA_API_KEY` — a different variable name — so `ChatOpenAI` found no key and sent unauthenticated requests.

**How it was fixed:**
Explicitly construct the `ChatOpenAI` instance with `api_key=self._api_key` and `base_url=<NIM endpoint>`, then pass it as `LLMRails(config, llm=llm)`. This bypasses the automatic env-var lookup entirely.

```python
llm = ChatOpenAI(
    model=self._model,
    api_key=self._api_key,
    base_url=self._base_url,
    temperature=0.4,
)
rails = LLMRails(config, llm=llm)
```

**Lesson:** Never assume LangChain adapters will pick up non-standard environment variable names. Always pass credentials explicitly when using third-party orchestration frameworks.

---

### Bug 2 — JSON Instruction Poisoned Phase 1 Intent Classification

**What failed:**
NeMo Guardrails Phase 1 (intent classification) returned `{"user_intent": "synthesise research gap"}` as the raw string — which was then parsed as intent `{`. Every request fell through to the generic fallback response regardless of what was submitted.

**Why it failed:**
`rails.yaml` had a global system instruction: `"You always respond with valid JSON only"`. This instruction applied to all phases, including Phase 1 (the internal intent classification step). The LLM dutifully returned JSON in Phase 1, but the Guardrails runtime expected a plain string like `"synthesise research gap"` — not a JSON object. The JSON-wrapped intent failed to match any defined Colang flow.

**How it was fixed:**
Removed the blanket JSON instruction from `rails.yaml`. JSON enforcement was moved to the Phase B synthesis prompt only, where it is passed directly to the NIM model without going through the Guardrails runtime.

**Lesson:** NeMo Guardrails processes multiple internal phases (intent classification, next-step generation, bot message generation). Global prompt instructions apply to all of them. Instructions intended only for the final output step must be isolated from the guardrail prompt context.

---

### Bug 3 — arXiv 429 Rate Limits (2+ Minute Delays)

**What failed:**
Paper retrieval regularly stalled for 2+ minutes, blocking the entire synthesis pipeline. The arXiv client's default configuration was triggering rate-limit throttling.

**Why it failed:**
`arxiv.Client()` defaults to `page_size=100`, causing every query to request `?max_results=100`. The arXiv API returned HTTP 429 (Too Many Requests) and the client's built-in retry mechanism waited through three exponential backoff cycles before giving up or succeeding.

**How it was fixed:**
Replaced arXiv with a two-source strategy: Semantic Scholar as primary, OpenAlex as fallback. Both return structured abstracts without the aggressive rate limits of the arXiv client.

**Lesson:** arXiv's Python client is designed for bulk batch downloads, not real-time retrieval. For low-latency use cases, OpenAlex or Semantic Scholar are better-suited alternatives.

---

### Bug 4 — Semantic Scholar API Key 403

**What failed:**
Semantic Scholar returned HTTP 403 on all endpoints despite having a valid-looking API key.

**Why it failed:**
The Semantic Scholar API key (`30eKamZKNg8itK6Rmg7Zx7cKTK8BupZ64kzfw89Y`) had been issued but not yet activated on Semantic Scholar's backend. Newly issued keys have an activation delay.

**How it was fixed:**
Switched the primary paper retrieval source to OpenAlex. OpenAlex is free, requires no API key, and covers cross-disciplinary research. Semantic Scholar was kept as a secondary fallback in the codebase for future use once the key activates.

**Lesson:** Always verify API key activation before building retrieval logic around a new provider. Having a no-auth fallback (OpenAlex) as the primary source eliminates this class of failure entirely.

---

### Bug 5 — Brev Instance Stuck on CPU-Only Node

**What failed:**
The initial Brev instance could not run `brev shell`. The control plane kept returning TLS timeout errors. The instance also had no GPU, which blocked any attempt to run a local NIM container.

**Why it failed:**
The instance was provisioned as `n2d-standard-4` — a CPU-only GCP machine type. Brev's control plane was experiencing intermittent TLS connectivity issues that prevented `brev shell` from establishing a connection.

**How it was fixed:**
1. Deleted the CPU instance.
2. Provisioned a new `litsynth-demo` instance with `g2-standard-4:nvidia-l4:1` (NVIDIA L4 GPU).
3. Bypassed `brev shell` entirely: retrieved the instance IP from the Brev API and connected directly via `ssh -i ~/.brev/brev.pem ubuntu@<ip>`.

**Lesson:** `brev shell` is a convenience wrapper; direct SSH is always available as a fallback. When control plane issues block the managed connection flow, the instance IP and the `~/.brev/brev.pem` key provide a reliable fallback path.

---

## Phase 2: NeMo Guardrails Pipeline Failures

The second phase moved synthesis from a direct NIM call to a full NeMo Guardrails pipeline. Three structural pipeline bugs emerged.

---

### Bug 6 — Phase 2 Always Returns `general response`

**What failed:**
After passing Phase 1 intent classification, Phase 2 (`generate_next_steps`) always selected the `general response` bot action regardless of the topic. The synthesis flow was never triggered.

**Why it failed:**
The `research synthesis guardrail` flow was registered only under `rails.output.flows`. Output rails are evaluated in Phase 3 (after the bot message is generated), not in Phase 2 (next-step selection). Phase 2 only saw the `handle off topic` example — a single negative example — and with no positive synthesis examples to match against, it defaulted to `general response` for every input.

**How it was fixed:**
1. Removed `research synthesis guardrail` from `rails.output.flows`.
2. Added `define user synthesise research gap` in `synthesiser.co` with representative examples matching the labels that LLaMA 70B actually produces during Phase 1 classification.
3. Added a `synthesise and propose flow` that maps both intent variants (`synthesise research gap`, `research synthesis`) to `bot respond`.

**Lesson:** NeMo Guardrails' phases are not interchangeable. Output flows do not participate in next-step selection. Synthesis flows must be registered as intent → bot action mappings, not output rails.

---

### Bug 7 — Phase 3 Generated Conversational Prose Instead of JSON

**What failed:**
Phase 3 (`generate_bot_message`) returned natural language explanations instead of JSON. When the response did contain JSON, it was truncated mid-structure with `Unterminated string` parse errors.

**Why it failed:**
Two compounding issues:
1. NeMo Guardrails Phase 3 is designed to produce natural language bot responses. It receives the synthesis prompt and rewrites it into conversational output — destroying the JSON structure.
2. The Colang runtime consumed ~100 tokens of context before the synthesis content, leaving fewer than 924 tokens for the model's response. Complex JSON outputs were truncated at the 1024-token limit.

**How it was fixed:**
Adopted a two-phase strategy that separates the guardrail concern from the inference concern:
- **Phase A**: A lightweight call to `LLMRails` with only the topic string. Used exclusively for intent classification (pass/refuse). No synthesis happens here.
- **Phase B**: A direct `httpx` call to `https://inference.local/v1/chat/completions` with the full synthesis prompt, bypassing Phase 3 entirely. The full 1024-token budget is available for the JSON output.

**Lesson:** NeMo Guardrails' Phase 3 is a natural language generator, not a JSON passthrough. Any application requiring structured JSON output from a NIM model must bypass Phase 3 and call the model directly, using Guardrails only for the intent classification step.

---

### Bug 8 — JSON Parser Failed on Preamble Text

**What failed:**
Even when the NIM model returned valid JSON, the parser raised `json.JSONDecodeError` on strings like `"Here is the analysis:\n{...}"`.

**Why it failed:**
The original `_parse_and_validate()` used `response.find("{")` to locate the start of the JSON object but did not correspondingly locate the end. When the model prepended preamble text, `find("{")` correctly found the opening brace but the subsequent `json.loads()` call included trailing text after the closing `}`, or — when the JSON was truncated — included only a partial structure.

**How it was fixed:**
Slice the response from `first "{"` to `last "}"` before parsing:

```python
start = response.find("{")
end = response.rfind("}") + 1
if start == -1 or end == 0:
    raise ValueError("No JSON object found in response")
json.loads(response[start:end])
```

**Lesson:** LLMs frequently prepend or append explanatory text even when instructed to return JSON only. Always bracket-search for both the opening and closing delimiters before parsing, and never assume the raw response string is pure JSON.

---

## Phase 3: NeMoClaw Discord Agent Setup

The third phase replaced the FastAPI interface with a Discord bot orchestrated through NeMoClaw. Ten infrastructure bugs were hit during sandbox setup and execution.

---

### Bug 9 — `skill install` Failed: No SKILL.md Found

**What failed:**
`nemoclaw litsynth-sandbox skill install skills/litsynth` exited with an error indicating no skill manifest was found.

**Why it failed:**
NeMoClaw requires a `SKILL.md` file at the root of the skill directory. The file must contain valid YAML frontmatter with specific required fields. The directory existed but had no manifest.

**How it was fixed:**
Created `skills/litsynth/SKILL.md` with the required frontmatter:

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

**Lesson:** NeMoClaw skill installation is manifest-driven. The `SKILL.md` file is not optional documentation — it is a required machine-readable contract for the install process.

---

### Bug 10 — `policy-add` Failed: `preset.name` Missing

**What failed:**
`nemoclaw litsynth-sandbox policy-add --from-file config/openshell_policy.yaml` returned a validation error about a missing `preset.name` field.

**Why it failed:**
The initial policy YAML used a top-level `name:` key at the root level instead of `preset.name:`. NeMoClaw expects the name to be nested under the `preset` key.

**How it was fixed:**
Changed the YAML structure from:

```yaml
name: litsynth-policy
```

to:

```yaml
preset:
  name: litsynth-policy
```

**Lesson:** NeMoClaw policy files have a specific schema. The `preset.name` field is required and must use RFC 1123 label format (lowercase letters, digits, hyphens; no underscores, no dots).

---

### Bug 11 — `policy-add` Failed: Unknown Field `default`

**What failed:**
After fixing the `preset.name` issue, `policy-add` failed again with an unknown field error pointing to `network_policies.egress.default`.

**Why it failed:**
The policy YAML included `default: deny` under the egress block, intending to enforce a deny-all baseline with an explicit allow-list. This field does not exist in the NeMoClaw policy schema — only `name`, `endpoints`, and `binaries` are valid fields under `preset`.

**How it was fixed:**
Removed `default: deny`. Deny-by-default is the implicit behaviour of the OpenShell proxy — there is no field needed to enable it.

**Lesson:** OpenShell's proxy is implicitly deny-by-default. Attempting to specify this explicitly causes a schema validation error. Only the allow-list needs to be declared.

---

### Bug 12 — `policy-add` Failed: Endpoints Expected Struct, Not String

**What failed:**
After fixing the `default` field issue, `policy-add` failed with a type mismatch error: endpoints were expected to be objects but were strings.

**Why it failed:**
The endpoints list was written as plain strings:

```yaml
endpoints:
  - api.openalex.org
  - inference.local
```

NeMoClaw expects each endpoint to be a struct with at minimum a `host` key.

**How it was fixed:**
Changed to struct format:

```yaml
endpoints:
  - host: api.openalex.org
  - host: inference.local
  - host: discord.com
  - host: gateway.discord.gg
```

**Lesson:** NeMoClaw policy endpoint entries are typed objects, not bare strings. This allows the schema to support additional fields (ports, protocols) in future versions.

---

### Bug 13 — Skill Workdir `/workspace` Does Not Exist

**What failed:**
The NeMoClaw exec command failed immediately with a `no such directory` error when targeting `/workspace` as the working directory.

**Why it failed:**
The initial `bot.py` assumed the skill workdir would be `/workspace` — a common convention in containerised environments. NeMoClaw installs skills to a different path inside the OpenShell container.

**How it was discovered:**
```bash
nemoclaw litsynth-sandbox exec --no-tty -- find /sandbox -name "synthesise.py"
# Output: /sandbox/.openclaw/skills/litsynth/synthesise.py
```

**How it was fixed:**
Updated `bot.py`:

```python
SKILL_WORKDIR = "/sandbox/.openclaw/skills/litsynth"
```

**Lesson:** Never assume skill installation paths in sandbox environments. Use `find` to discover the actual path after installation. NeMoClaw installs skills under `/sandbox/.openclaw/skills/<skill-name>/`.

---

### Bug 14 — `python: command not found` in Sandbox

**What failed:**
The exec command succeeded but the skill failed to start:
```
/bin/sh: python: command not found
```

**Why it failed:**
The sandbox base image only includes `python3`, not a `python` symlink. The subprocess command in `bot.py` used `python` (no version suffix).

**How it was fixed:**
Changed `python` to `python3` everywhere in the exec command construction.

**Lesson:** Always use `python3` explicitly in cross-environment scripts. The `python` → `python3` symlink is not guaranteed in minimal container images.

---

### Bug 15 — `ModuleNotFoundError: No module named 'httpx'`

**What failed:**
`synthesise.py` crashed on import with `ModuleNotFoundError: No module named 'httpx'`.

**Why it failed:**
While `httpx` was listed in `skills/litsynth/requirements.txt`, the `skill install` command did not automatically install Python dependencies into the sandbox's Python environment.

**How it was fixed:**
Manually installed the dependency inside the sandbox:

```bash
nemoclaw litsynth-sandbox exec --no-tty -- \
  pip3 install --break-system-packages httpx
```

The `--break-system-packages` flag was needed because the sandbox Python was a system install (not a venv).

**Lesson:** NeMoClaw `skill install` copies skill files into the sandbox but does not automatically run `pip install -r requirements.txt`. Dependencies must be installed separately, or the skill entrypoint must handle its own dependency installation at startup.

---

### Bug 16 — Custom Policy `litsynth-policy` Not Active on Gateway

**What failed:**
`nemoclaw litsynth-sandbox policy-list` consistently showed:

```
○ litsynth-policy — custom preset (recorded locally, not active on gateway)
```

All outbound connections to `api.openalex.org` returned `403 Connection Tunnel Error`. The custom policy had no effect on the actual proxy.

**Why it failed:**
In NeMoClaw v0.0.46, `--from-file` policy presets are stored in the local NeMoClaw config but are not propagated to the OpenShell gateway proxy container. Only built-in named presets (e.g., `discord`) are applied at the proxy level. This appears to be a known limitation of v0.0.46's policy propagation mechanism.

**How it was addressed:**
The limitation could not be worked around within the sandbox. The architecture was redesigned to move paper retrieval to the host (where network is unrestricted), eliminating the need for `api.openalex.org` access from inside the sandbox. See Bug 18 for the full resolution.

**Lesson:** In NeMoClaw v0.0.46, custom file-based policies are not reliably gateway-active. Architecture decisions should account for the possibility that only built-in presets are available for sandbox egress control. Design the sandbox to do only what the Privacy Router allows (`inference.local`), and handle external API calls on the unrestricted host.

---

### Bug 17 — 403 on `integrate.api.nvidia.com` from Inside Sandbox

**What failed:**
NIM API calls from inside the sandbox returned `403 Forbidden`. The proxy was blocking connections to `integrate.api.nvidia.com`.

**Why it failed:**
Same root cause as Bug 16: the custom policy that would have allowed `integrate.api.nvidia.com` was not gateway-active. Additionally, `integrate.api.nvidia.com` requires NVIDIA credentials which were not available inside the sandbox.

**How it was discovered:**
Testing individual hostnames via the proxy revealed that `inference.local:443` returns `200 Connection Established` — it is handled by the OpenShell Privacy Router, which is a built-in routing rule, not a policy-based allow. The Privacy Router intercepts `inference.local`, strips the placeholder credential, injects the real `NVIDIA_API_KEY` from the host credential store, and forwards to `integrate.api.nvidia.com`.

**How it was fixed:**
Changed `NIM_BASE_URL` from `https://integrate.api.nvidia.com/v1` to `https://inference.local/v1` inside the sandbox. Set `api_key = "openshell-managed"` as the placeholder.

```python
# synthesise.py (inside sandbox)
NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "https://inference.local/v1")
headers = {
    "Authorization": "Bearer openshell-managed",
    "Content-Type": "application/json",
}
```

**Lesson:** `inference.local` is the correct NIM inference endpoint from inside an OpenShell sandbox. It is not a custom configuration — it is the OpenShell Privacy Router's built-in intercept rule. Never attempt to call `integrate.api.nvidia.com` directly from inside the sandbox.

---

### Bug 18 — OpenAlex Still Blocked After `inference.local` Fix

**What failed:**
NIM inference was now working, but paper retrieval from `api.openalex.org` still returned `403 Forbidden` from the proxy. The full synthesis pipeline could not complete.

**Why it failed:**
Same root cause as Bug 16. The `litsynth-policy` allowing `api.openalex.org` was not gateway-active. There was no built-in Privacy Router rule for OpenAlex equivalent to `inference.local`.

**How it was fixed:**
Architectural split of responsibilities between host and sandbox:

```
BEFORE:
  bot.py (host)     → nemoclaw exec synthesise.py <topic>
  synthesise.py     → GET api.openalex.org  ❌ blocked
                    → POST inference.local  ✓

AFTER:
  bot.py (host)     → GET api.openalex.org  ✓
                    → nemoclaw exec synthesise.py <topic> \
                         --context <papers_json>
  synthesise.py     → parse --context arg   ✓
                    → POST inference.local  ✓
```

`bot.py` now:
1. Fetches 4 paper abstracts from OpenAlex (unrestricted host network)
2. Serialises them as JSON
3. Passes them to `synthesise.py` via the `--context` command-line argument

`synthesise.py` now:
1. Parses `--context` argument instead of fetching papers
2. Calls only `inference.local` — the one endpoint reliably available

**Lesson:** When a sandbox's network policy cannot be made permissive enough for all required services, split the work across the network boundary. Put external API calls on the unrestricted host, and keep only managed-credential inference inside the sandbox. This is the correct long-term architecture regardless of policy limitations.

---

## Key Architectural Decisions

### Decision 1: Discord over FastAPI

The FastAPI REST interface was decommissioned in favour of a Discord bot for several reasons:
- Discord provides a conversational UI that matches the use case (researchers asking questions, getting answers)
- The bot token model eliminates the need for authentication middleware on the backend
- Discord handles rate limiting, message formatting, and file upload naturally
- The `!synthesize` command pattern is more intuitive than a REST endpoint for a research tool

The FastAPI code was retained (not deleted) because the DB models, prompt templates, and NeMoClaw client may be reused in a future web interface.

### Decision 2: Two-Phase NeMo Guardrails

Running Guardrails for off-topic detection only (Phase A) and making a direct NIM call for synthesis (Phase B) was chosen over alternatives:

| Approach | Problem |
|---|---|
| Single LLMRails call for everything | Phase 3 rewrites JSON to prose; token budget wasted on Colang overhead |
| No Guardrails at all | No off-topic protection; anything gets synthesised |
| Output rails for JSON enforcement | Output rails invoke Phase 3 — same prose-rewriting problem |
| Two-phase split (chosen) | Clean separation: Guardrails for intent, direct NIM for output |

### Decision 3: Host-Fetched Papers

Moving paper retrieval to the host was a direct response to Bug 16/18 (custom policies not gateway-active). This turned out to produce a better architecture regardless:
- Failures in paper retrieval are isolated to `bot.py` and never corrupt the sandbox environment
- The sandbox becomes stateless between invocations — it receives all inputs via arguments, not network calls
- Fetching on the host allows for future caching, deduplication, and source switching without touching sandbox code

### Decision 4: `inference.local` as the NIM Endpoint

The Privacy Router pattern — where the sandbox calls a local hostname and the router injects credentials — is the correct security model for sandboxed AI inference. It means:
- The sandbox never holds real NVIDIA credentials
- Compromising the sandbox does not expose API keys
- The credential injection point (the router) is outside the trust boundary of the code being executed

### Decision 5: Direct SSH Fallback for Brev

Discovering that `brev shell` is a wrapper around SSH (not a proprietary tunnel) and that the `~/.brev/brev.pem` key works directly with `ssh -i` was critical for maintaining development velocity when the Brev control plane had TLS issues. Always identify the direct-access fallback for any managed development environment.

---

## What We Learned About NeMoClaw and OpenShell Internals

### NeMoClaw v0.0.46 limitations

1. **Custom `--from-file` policies are not gateway-active.** They are recorded in the local NeMoClaw config but not propagated to the OpenShell proxy container. This is the single most significant undocumented limitation encountered.

2. **Built-in presets (`discord`) are the only reliably gateway-active policies.** If your sandbox needs to reach an external service, check whether a built-in preset covers it before building custom policy logic.

3. **`inference.local` is a first-class router rule, not a policy entry.** It is always available from inside an OpenShell sandbox and is the canonical way to reach NIM inference. It is not documented in the policy YAML — it is baked into the router.

4. **Skill manifest is required, not optional.** `SKILL.md` with correct YAML frontmatter is mandatory for `skill install` to succeed. The `name`, `version`, `description`, `entrypoint`, `dependencies`, and `env` fields must all be present.

5. **`skill install` does not run `pip install`.** Dependencies listed in `requirements.txt` are not automatically installed. A separate exec command is needed.

6. **Sandbox Python is `python3`, not `python`.** The `python` symlink does not exist in the sandbox base image.

### NeMo Guardrails internals

1. **Phase 1 (intent classification) is a separate LLM call.** Global prompt instructions apply to it. Instructions like "respond only in JSON" cause Phase 1 to return JSON-wrapped intents, which the Guardrails runtime cannot parse.

2. **Phase 3 (bot message generation) is a paraphrasing step.** Its job is to generate natural language. Registering a synthesis flow as an output rail causes Phase 3 to rewrite structured JSON output into prose.

3. **Colang flow registration location matters.** Input flows participate in Phase 1/2 (intent + next-step). Output flows participate in Phase 3 (message generation). For off-topic detection, use input flows. For JSON output generation, bypass Phase 3 entirely.

4. **Token budget is shared across all phases.** With `max_tokens: 1024`, ~100 tokens consumed by Phase 1/2 context leaves ~924 tokens for Phase 3 output. Complex JSON structures at the token limit will be truncated.

### OpenShell network model

1. **The proxy is deny-by-default.** No egress is allowed unless explicitly permitted by an active policy or a built-in router rule.

2. **`10.200.0.1:3128` is always the proxy address.** This is the OpenShell gateway container's internal IP, consistent across sandbox instances.

3. **`inference.local` intercept is a router-level rule.** It is not a policy entry and cannot be disabled or modified by policy files.

4. **Custom policy propagation is unreliable in v0.0.46.** Budget for this limitation when designing any sandbox architecture that requires external API access beyond the built-in presets.

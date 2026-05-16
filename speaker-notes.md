# LitSynth — Speaker Notes
**Architecture Walkthrough · 5–10 minutes**

---

## Opening (30 sec)

> "So this is LitSynth — a backend research synthesis API I built to demonstrate autonomous AI workflow orchestration using NVIDIA's inference stack.
>
> The core idea is simple: you give it a research topic, and it autonomously retrieves relevant literature, identifies the research gap across those papers, and returns a structured, database-persisted experiment hypothesis — all through a REST API, no human in the loop after the initial request.
>
> Let me walk you through the architecture layer by layer."

---

## Layer 1 — Client Layer (45 sec)

*Point to the top blue bar.*

> "At the top we have the client layer. Right now, this is Swagger UI — FastAPI auto-generates a full interactive API interface at `/docs`, so the entire system is testable without any frontend.
>
> This was a deliberate choice. I wanted to demonstrate backend engineering capability, not UI work. Any frontend — React, Next.js, whatever — can be bolted on later by consuming the REST API. The backend doesn't care.
>
> The client sends standard HTTP requests. Nothing special here — this boundary is clean and well-defined."

---

## Layer 2 — FastAPI Router (1 min)

*Point to the green endpoint boxes.*

> "Below that is the FastAPI router layer. There are five main endpoints.
>
> The most important one for the demo is `POST /analyze`. You submit a research topic — it returns a `task_id` and a `202 Accepted` immediately. The pipeline doesn't block the HTTP response. That's intentional — LLM inference takes anywhere from five to thirty seconds depending on load. Holding an HTTP connection open for that long is fragile. Load balancers time out, clients disconnect. So I decouple the request from the computation right here.
>
> Then you poll `GET /task/{id}` to track the status. You watch it transition through states in real time. Once it hits `COMPLETED`, you call `/results` to fetch the hypothesis. And `/logs` gives you the full audit trail of everything the agent did.
>
> On the far right there's the `BackgroundTasks` box — FastAPI's built-in mechanism for running work off the request thread. Each background job gets its own database session so there's no shared state across threads."

---

## Layer 3 — Agent Pipeline (2 min)

*Point to the amber left panel.*

> "This is the heart of the system — the agent pipeline, implemented as a state machine.
>
> Every research request goes through four stages: PENDING, RETRIEVING, SYNTHESIZING, and then either COMPLETED or FAILED.
>
> PENDING is just the initial state — the task row has been created in the database, the background thread is starting up.
>
> RETRIEVING is where the system fetches literature context. Right now this is backed by a curated mock corpus — four research areas, three to four realistic paper abstracts each. The topic gets keyword-matched to the closest corpus and the abstracts are formatted into a structured context block. In the planned production version, this slot gets replaced by a real arXiv or Semantic Scholar API call. The interface doesn't change — just the implementation behind it.
>
> SYNTHESIZING is where the LLM call happens. The system compiles the prompt — system instructions plus the literature context — and sends it to the inference layer.
>
> And then it either lands on COMPLETED with a persisted hypothesis, or FAILED with an error message stored in the database. The failure path is important — every exception is caught at the top level so the task never hangs silently in SYNTHESIZING. If something goes wrong, you can always find out what and when."

---

## Layer 3 — NeMoClaw Orchestration Zone (2 min)

*Point to the green right panel.*

> "On the right side of that same row is the NeMoClaw orchestration layer — this is where the actual AI inference happens.
>
> Let me break this into three parts.
>
> First, the prompt templates. I keep these isolated in their own module — `prompts/research_synthesis.py`. The system prompt sets the persona and enforces a strict JSON output rule. The user template injects the dynamic variables — topic and literature context. Keeping prompts separate from orchestration logic means I can iterate on them independently without touching the agent runner.
>
> Second, the `NeMoClawClient`. This is my adapter layer between FastAPI and NVIDIA's NIM inference API. It does three things beyond a plain HTTP call: it compiles the prompt, it calls the NIM endpoint using `httpx`, and — this is the key part — it validates the response against a Pydantic schema called `HypothesisOutput`. If the LLM returns malformed JSON or missing fields, the client automatically retries up to two times before raising an error.
>
> Third, NVIDIA NIM itself. Right now I'm calling the hosted NIM API — `integrate.api.nvidia.com` — using `meta/llama-3.1-70b-instruct`. It's an OpenAI-compatible endpoint, so the integration is clean. The planned upgrade is to point this at a local NIM container running on a Brev GPU instance, which is where NeMoClaw's native SDK comes in."

---

## HypothesisOutput Schema (45 sec)

*Point to the purple schema box.*

> "This is the structured output contract — and I think it's one of the most important architectural decisions in the whole system.
>
> The LLM is not trusted to return freeform text. The system prompt explicitly tells it to return only a JSON object matching this schema: gap identified, proposed architecture, evaluation metric, and confidence score.
>
> The same Pydantic model that validates the LLM output also drives the database write. So there's no parsing layer between inference and persistence — if the object validates, it goes straight into the `generated_hypotheses` table. This is the difference between a demo chatbot and an engineered pipeline."

---

## Layer 4 — Persistence (1 min)

*Point to the purple bottom layer.*

> "At the bottom is the persistence layer — SQLite backed by SQLAlchemy ORM.
>
> Three tables. `research_tasks` is the control record — it owns the state machine, stores the error message if something fails, and timestamps every update. `generated_hypotheses` is the output record — it only exists when the pipeline completes successfully, which is a meaningful signal. And `agent_run_logs` is an append-only audit trail — every state transition writes a row, with the stage name, a message, and a timestamp.
>
> SQLite was chosen purely for development speed — zero infrastructure, single file. The SQLAlchemy abstraction means switching to PostgreSQL for a real deployment is literally one line in the environment config.
>
> The reason I care about persistence beyond just storing results is traceability. In an autonomous system, you need to be able to answer: what did the agent do, in what order, and why did it succeed or fail? The log table answers that without needing to scrape application logs."

---

## Closing — Planned Path to Brev + NeMoClaw (1 min)

> "So to summarise the current state — the full pipeline works locally. The LLM calls go out to NVIDIA's hosted NIM API, the structured output is validated, persisted to SQLite, and exposed through the REST API. Everything is testable through Swagger UI.
>
> The planned path from here is straightforward. On Brev, I'd provision a GPU instance, run a local NIM container, and change one environment variable to point the inference layer at it. The `NeMoClawClient` adapter then gets its `_call_nim` method replaced with native NeMoClaw SDK calls — the rest of the system is untouched. Brev's port-forwarding then exposes the FastAPI server locally for demo access.
>
> The architecture was designed with this transition in mind. The inference backend is completely isolated behind the client interface. Swapping it doesn't cascade into the agent runner, the database, or the API layer."

---

## Likely Questions & Answers

**"Why not just use LangChain?"**
> "LangChain is an abstraction layer that helps you chain LLM calls quickly. But it adds a dependency that obscures what's actually happening — and in a demo, you want an evaluator to see the system thinking, not a library's internals. Building the orchestration directly makes the architecture visible and defensible."

**"Is this actually autonomous?"**
> "Autonomous in the sense that matters for this context: zero human intervention after the initial request. The pipeline transitions states, calls the LLM, validates and retries on failure, and persists results — entirely on its own. The retrieval layer is mocked, which I'm upfront about, but that's a data sourcing decision, not an autonomy one."

**"How would you scale this?"**
> "Three changes: swap SQLite for PostgreSQL, replace FastAPI BackgroundTasks with Celery and Redis for a proper task queue, and run multiple uvicorn workers behind a load balancer. The core pipeline logic doesn't change — it's already stateless and session-isolated per run."

**"What does NeMoClaw specifically add?"**
> "In the current implementation, `NeMoClawClient` is my adapter layer that sits where NeMoClaw would natively sit — it handles prompt compilation, inference requests, and schema enforcement. The native NeMoClaw SDK would replace the raw HTTP call in `_call_nim()` and add first-class support for tool binding and guardrails. The interface I've built is designed to accept that replacement cleanly."

**"Why mock the retrieval instead of using real arXiv?"**
> "Scraping PDFs introduces network failures, encoding edge cases, and rate limits — all of which would consume more time to handle gracefully than the retrieval itself adds in demo value. The mock produces the same quality of LLM input. The retrieval slot is architecturally isolated, so replacing it is one function in `mock_data.py`."

---

## Timing Guide

| Section | Target Time |
|---|---|
| Opening | 0:30 |
| Client layer | 1:15 |
| FastAPI router | 2:15 |
| Agent pipeline | 4:15 |
| NeMoClaw zone | 6:15 |
| Schema | 7:00 |
| Persistence | 8:00 |
| Closing + Brev path | 9:00 |
| Buffer for questions | 9:00–10:00 |

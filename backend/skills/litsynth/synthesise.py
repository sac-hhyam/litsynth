#!/usr/bin/env python3
"""
LitSynth Synthesis Skill — runs INSIDE the NeMoClaw OpenShell sandbox.

Invoked by bot.py on the host via:
    nemoclaw lit-synth-sandbox exec --no-tty --workdir /workspace -- python synthesise.py <topic>

Stdout is the only communication channel back to the host process.
The script emits a single JSON object on stdout, then exits.

Environment variables expected inside the sandbox:
    NVIDIA_API_KEY   — NVIDIA NIM API key
    NIM_BASE_URL     — defaults to https://integrate.api.nvidia.com/v1
    NIM_MODEL        — defaults to nvidia/nemotron-3-nano-omni-30b-a3b-reasoning

Install as a skill:
    nemoclaw lit-synth-sandbox skill install backend/skills/litsynth
"""
from __future__ import annotations

import json
import logging
import os
import sys
import textwrap

import httpx

# ── Config (from sandbox env) ─────────────────────────────────────────────────
NVIDIA_API_KEY = "openshell-managed"   # credentials injected by OpenShell Privacy Router
NIM_BASE_URL   = os.environ.get("NIM_BASE_URL", "https://inference.local/v1")
NIM_MODEL      = os.environ.get("NIM_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
MAX_TOKENS     = int(os.environ.get("MAX_SYNTHESIS_TOKENS", "2048"))
TEMPERATURE    = float(os.environ.get("LLM_TEMPERATURE", "0.4"))
TIMEOUT        = 60.0

# OpenShell Privacy Router intercepts inference.local:443 and forwards to
# NVIDIA's cloud with injected credentials. It re-signs TLS with its own CA —
# point httpx to that bundle so SSL verification passes.
_OPENSHELL_CA  = "/etc/openshell-tls/ca-bundle.pem"
SSL_VERIFY: str | bool = (
    _OPENSHELL_CA if os.path.exists(_OPENSHELL_CA) else True
)

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger(__name__)


# ── Paper retrieval (OpenAlex) ────────────────────────────────────────────────

def _reconstruct_abstract(inv_idx: dict | None) -> str:
    """Convert OpenAlex inverted-index abstract to plain text."""
    if not inv_idx:
        return ""
    max_pos = max(pos for positions in inv_idx.values() for pos in positions)
    tokens: list[str] = [""] * (max_pos + 1)
    for word, positions in inv_idx.items():
        for pos in positions:
            tokens[pos] = word
    return " ".join(t for t in tokens if t)


def fetch_papers(topic: str, max_results: int = 4) -> list[dict]:
    """Fetch papers from OpenAlex. Raises RuntimeError on failure."""
    params = {
        "search":   topic,
        "per_page": max_results,
        "select":   "display_name,authorships,abstract_inverted_index,publication_year",
        "mailto":   "litsynth@demo.nvaitc.ai",
    }
    with httpx.Client(timeout=TIMEOUT, verify=SSL_VERIFY) as client:
        resp = client.get("https://api.openalex.org/works", params=params)
    resp.raise_for_status()

    papers = []
    for w in resp.json().get("results", []):
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
        if not abstract.strip():
            continue
        names = [
            a["author"]["display_name"]
            for a in w.get("authorships", [])
            if a.get("author", {}).get("display_name")
        ]
        author_str = (", ".join(names[:3]) + " et al." if len(names) > 3
                      else ", ".join(names))
        year = w.get("publication_year")
        if year:
            author_str = f"{author_str}, {year}"
        papers.append({
            "title":    w.get("display_name", "Untitled"),
            "authors":  author_str,
            "abstract": textwrap.shorten(abstract, width=800, placeholder="..."),
        })

    if not papers:
        raise RuntimeError(f"OpenAlex returned no papers with abstracts for: '{topic}'")
    return papers


def format_context(papers: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(papers, start=1):
        blocks.append(
            f"[PAPER {i}]\nTitle: {p['title']}\nAuthors: {p['authors']}\n"
            f"Abstract:\n{p['abstract']}"
        )
    return "\n\n---\n\n".join(blocks)


# ── NIM synthesis ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a Senior AI Research Scientist specialising in identifying research gaps \
and proposing novel experiment architectures.
Your ENTIRE response must be a single valid JSON object. No preamble, no explanation outside JSON.
Schema:
{
  "gap_identified": "<specific, evidence-backed limitation shared across papers>",
  "proposed_architecture": "<named architecture with components, data flow, training paradigm>",
  "evaluation_metric": "<metric on benchmark>",
  "confidence_score": "<LOW|MEDIUM|HIGH>"
}
Do NOT include markdown fences, extra keys, or trailing text."""

USER_TEMPLATE = """\
Research Topic: {topic}

Literature Survey:
{context}

Synthesise the gap, propose a novel architecture, define the evaluation metric, \
and assign a confidence score. Reply with ONLY valid JSON matching the schema."""


def call_nim(topic: str, context: str) -> str:
    """Call NVIDIA NIM and return the raw completion text."""
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not set in the sandbox environment.")

    payload = {
        "model":       NIM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(topic=topic, context=context)},
        ],
        "max_tokens":  MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type":  "application/json",
    }
    with httpx.Client(timeout=TIMEOUT, verify=SSL_VERIFY) as client:
        resp = client.post(f"{NIM_BASE_URL}/chat/completions",
                           json=payload, headers=headers)
    if not resp.is_success:
        raise RuntimeError(f"NIM {resp.status_code}: {resp.text[:400]}")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_json(raw: str) -> dict:
    """Strip preamble/markdown fences and parse JSON."""
    import re
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    text = fence.group(1) if fence else raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    Usage:
      synthesise.py <topic>
          Fetch papers from OpenAlex then synthesise (requires network access).

      synthesise.py --context '<json>' <topic>
          Accept pre-fetched papers as JSON string — used when the sandbox
          proxy blocks OpenAlex. bot.py fetches papers on the host and passes
          them here so only the NIM call runs inside the sandbox.
    """
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: synthesise.py [--context <json>] <topic>"}))
        sys.exit(1)

    # Parse optional --context argument
    pre_fetched_papers: list[dict] | None = None
    args = sys.argv[1:]
    if args[0] == "--context":
        if len(args) < 3:
            print(json.dumps({"error": "--context requires a JSON argument and topic"}))
            sys.exit(1)
        try:
            pre_fetched_papers = json.loads(args[1])
        except json.JSONDecodeError as exc:
            print(json.dumps({"error": f"Invalid --context JSON: {exc}"}))
            sys.exit(1)
        args = args[2:]

    topic = " ".join(args)

    try:
        if pre_fetched_papers is not None:
            papers = pre_fetched_papers
            source = "host-fetched"
        else:
            papers = fetch_papers(topic)
            source = "openalex"

        context = format_context(papers)
        raw     = call_nim(topic, context)
        result  = extract_json(raw)
    except Exception as exc:
        print(json.dumps({"error": str(exc), "topic": topic}))
        sys.exit(1)

    result["topic"]       = topic
    result["papers_used"] = len(papers)
    result["source"]      = source

    # Emit the JSON result to stdout — bot.py reads this
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

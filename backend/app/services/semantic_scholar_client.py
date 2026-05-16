"""
Academic paper retrieval client.

Primary source: OpenAlex (https://openalex.org)
  - Free, no key required, excellent cross-disciplinary coverage
  - Polite pool: adds mailto= param to identify the caller
  - Returns clean titles, authors, and abstracts for any topic
  - Abstract stored as inverted-index; reconstructed to plain text here

Optional upgrade: Semantic Scholar Graph API
  - Set S2_API_KEY in .env to enable (1 req/s authenticated tier)
  - Falls back to OpenAlex automatically if S2 key is not set or returns an error

Both sources return dicts matching the shared corpus schema:
  {"title": str, "authors": str, "abstract": str}

Usage:
    papers = fetch_papers(topic, max_results=4)
    # raises RuntimeError if both sources fail
"""
from __future__ import annotations

import logging
import textwrap

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_TIMEOUT    = 15.0
_OA_BASE    = "https://api.openalex.org"
_S2_BASE    = "https://api.semanticscholar.org/graph/v1"
_OA_MAILTO  = "litsynth@demo.nvaitc.ai"   # identifies us in OpenAlex polite pool
_OA_FIELDS  = "display_name,authorships,abstract_inverted_index,publication_year"
_S2_FIELDS  = "title,authors,abstract,year"


# ── OpenAlex helpers ──────────────────────────────────────────────────────────

def _reconstruct_abstract(inv_idx: dict | None) -> str:
    """Convert OpenAlex inverted-index abstract back to readable text."""
    if not inv_idx:
        return ""
    max_pos = max(pos for positions in inv_idx.values() for pos in positions)
    tokens: list[str] = [""] * (max_pos + 1)
    for word, positions in inv_idx.items():
        for pos in positions:
            tokens[pos] = word
    return " ".join(t for t in tokens if t)


def fetch_from_openalex(topic: str, max_results: int = 4) -> list[dict]:
    """
    Fetch papers from OpenAlex for *topic*.
    Raises RuntimeError if the request fails or returns no papers with abstracts.
    """
    params = {
        "search":   topic,
        "per_page": max_results,
        "select":   _OA_FIELDS,
        "mailto":   _OA_MAILTO,
    }
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(f"{_OA_BASE}/works", params=params)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"OpenAlex HTTP {exc.response.status_code} for '{topic}'"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"OpenAlex connection error: {exc}") from exc

    works = resp.json().get("results", [])
    papers = []
    for w in works:
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
        if not abstract.strip():
            continue  # skip papers with no usable abstract

        # Build author string from authorships list
        author_names = [
            a["author"]["display_name"]
            for a in w.get("authorships", [])
            if a.get("author", {}).get("display_name")
        ]
        author_str = (
            ", ".join(author_names[:3]) + " et al."
            if len(author_names) > 3
            else ", ".join(author_names)
        )
        year = w.get("publication_year")
        if year:
            author_str = f"{author_str}, {year}"

        papers.append({
            "title":   w.get("display_name", "Untitled"),
            "authors": author_str,
            "abstract": textwrap.shorten(abstract, width=800, placeholder="..."),
            "openalex_id": w.get("id", ""),
        })

    if not papers:
        raise RuntimeError(
            f"OpenAlex returned no papers with abstracts for: '{topic}'"
        )

    logger.info("OpenAlex returned %d papers for topic '%s'", len(papers), topic)
    return papers


# ── Semantic Scholar (optional, requires S2_API_KEY) ──────────────────────────

def fetch_from_s2(topic: str, max_results: int = 4) -> list[dict]:
    """
    Fetch papers from Semantic Scholar Graph API.
    Requires S2_API_KEY to be set in settings.
    Raises RuntimeError on auth failure (403/401) or network error.
    """
    settings = get_settings()
    if not settings.S2_API_KEY:
        raise RuntimeError("S2_API_KEY not configured — skipping Semantic Scholar.")

    headers = {"x-api-key": settings.S2_API_KEY}
    params  = {"query": topic, "limit": max_results, "fields": _S2_FIELDS}

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(f"{_S2_BASE}/paper/search", params=params, headers=headers)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Semantic Scholar HTTP {exc.response.status_code} for '{topic}': "
            f"{exc.response.text[:200]}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Semantic Scholar connection error: {exc}") from exc

    papers = []
    for p in resp.json().get("data", []):
        abstract = p.get("abstract") or ""
        if not abstract.strip():
            continue
        names = [a["name"] for a in p.get("authors", []) if a.get("name")]
        author_str = (
            ", ".join(names[:3]) + " et al." if len(names) > 3 else ", ".join(names)
        )
        year = p.get("year")
        if year:
            author_str = f"{author_str}, {year}"
        papers.append({
            "title":   p.get("title", "Untitled"),
            "authors": author_str,
            "abstract": textwrap.shorten(abstract, width=800, placeholder="..."),
            "s2_paper_id": p.get("paperId", ""),
        })

    if not papers:
        raise RuntimeError(f"S2 returned no papers with abstracts for: '{topic}'")

    logger.info("Semantic Scholar returned %d papers for topic '%s'", len(papers), topic)
    return papers


# ── Unified fetch: S2 → OpenAlex ─────────────────────────────────────────────

def fetch_papers(topic: str, max_results: int = 4) -> tuple[list[dict], str]:
    """
    Fetch papers using the best available source.

    Order of preference:
      1. Semantic Scholar (if S2_API_KEY is set)
      2. OpenAlex (free, no key required)

    Returns (papers, source_label) where source_label is "semantic_scholar"
    or "openalex".  Raises RuntimeError if both sources fail.
    """
    settings = get_settings()

    if settings.S2_API_KEY:
        try:
            return fetch_from_s2(topic, max_results), "semantic_scholar"
        except RuntimeError as exc:
            logger.warning(
                "S2 failed for '%s' (%s) — trying OpenAlex.", topic, exc
            )

    papers = fetch_from_openalex(topic, max_results)
    return papers, "openalex"

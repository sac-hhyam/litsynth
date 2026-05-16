"""
Semantic Scholar retrieval client.

Fetches real paper abstracts for any topic via the Semantic Scholar Graph API.
Requires an API key (set S2_API_KEY in .env) for the 1 req/s rate limit tier.
Without a key the API still works but is aggressively throttled.

Endpoint used:
  GET https://api.semanticscholar.org/graph/v1/paper/search
  ?query=<topic>&limit=<n>&fields=title,authors,abstract,year

Response schema:
  {"total": N, "data": [{"paperId": "...", "title": "...",
    "abstract": "...", "year": N,
    "authors": [{"name": "..."}, ...]}, ...]}

Usage:
    papers = fetch_from_s2(topic, max_results=4)
    # raises RuntimeError on failure
"""
from __future__ import annotations

import logging
import textwrap

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS  = "title,authors,abstract,year"
_TIMEOUT = 15.0   # seconds — S2 responds quickly; fail fast on network issues


def fetch_from_s2(topic: str, max_results: int = 4) -> list[dict]:
    """
    Search Semantic Scholar for *topic* and return up to *max_results* papers.

    Each returned dict matches the shared corpus schema:
      {"title": str, "authors": str, "abstract": str}

    Raises RuntimeError if the search fails or returns no usable results.
    """
    settings = get_settings()
    headers: dict[str, str] = {}
    if settings.S2_API_KEY:
        headers["x-api-key"] = settings.S2_API_KEY

    params = {
        "query": topic,
        "limit": max_results,
        "fields": _FIELDS,
    }

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            response = client.get(
                f"{_S2_BASE}/paper/search",
                params=params,
                headers=headers,
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Semantic Scholar HTTP {exc.response.status_code} for '{topic}': "
            f"{exc.response.text[:200]}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Semantic Scholar connection error: {exc}") from exc

    data = response.json()
    raw_papers = data.get("data", [])

    papers = []
    for p in raw_papers:
        abstract = p.get("abstract") or ""
        if not abstract.strip():
            # Skip papers with no abstract — they are useless for synthesis
            continue

        authors_raw = p.get("authors", [])
        names = [a["name"] for a in authors_raw if a.get("name")]
        author_str = (
            ", ".join(names[:3]) + " et al." if len(names) > 3 else ", ".join(names)
        )
        year = p.get("year")
        if year:
            author_str = f"{author_str}, {year}"

        papers.append({
            "title": p.get("title", "Untitled"),
            "authors": author_str,
            "abstract": textwrap.shorten(abstract, width=800, placeholder="..."),
            "s2_paper_id": p.get("paperId", ""),
        })

    if not papers:
        raise RuntimeError(
            f"Semantic Scholar returned no papers with abstracts for: '{topic}'"
        )

    logger.info(
        "Semantic Scholar returned %d papers for topic '%s'", len(papers), topic
    )
    return papers

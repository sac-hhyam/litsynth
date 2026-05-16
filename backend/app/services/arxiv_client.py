"""
arXiv retrieval client.

Fetches real paper abstracts for any topic using the arXiv API (no key required).
Falls back gracefully to the mock corpus if the search fails or returns nothing.

Usage:
    papers, source = get_papers(topic, max_results=4)
    # source is "arxiv" or "mock"

Performance notes:
  - page_size is set to match max_results so the API returns exactly the
    number of papers we need (not 100 by default), avoiding large payloads
    and reducing the chance of HTTP 429 rate-limit errors.
  - num_retries=1, delay_seconds=1 means we give up quickly and fall back
    to mock rather than burning ~2 minutes on retries.
"""
from __future__ import annotations

import logging
import textwrap

import arxiv

logger = logging.getLogger(__name__)


def fetch_from_arxiv(topic: str, max_results: int = 4) -> list[dict]:
    """
    Search arXiv for the topic and return a list of paper dicts
    matching the same schema as the mock corpus:
      {"title": str, "authors": str, "abstract": str}

    Raises RuntimeError if the search returns no results.
    """
    # Create a fresh client per call so page_size matches max_results exactly.
    # This keeps the API request small (e.g. max_results=4 → ?max_results=4)
    # and avoids the default page_size=100 that triggers 429 rate limits.
    client = arxiv.Client(
        page_size=max_results,
        num_retries=1,       # fail fast → quick mock fallback
        delay_seconds=1,
    )

    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for result in client.results(search):
        authors = [str(a) for a in result.authors]
        author_str = (
            ", ".join(authors[:3]) + " et al."
            if len(authors) > 3
            else ", ".join(authors)
        )
        papers.append({
            "title": result.title,
            "authors": author_str,
            "abstract": textwrap.shorten(result.summary, width=800, placeholder="..."),
            "arxiv_id": result.entry_id,
        })

    if not papers:
        raise RuntimeError(f"arXiv returned no results for query: '{topic}'")

    logger.info("arXiv returned %d papers for topic '%s'", len(papers), topic)
    return papers

"""
arXiv retrieval client.

Fetches real paper abstracts for any topic using the arXiv API (no key required).
Falls back gracefully to the mock corpus if the search fails or returns nothing.

Usage:
    papers, source = get_papers(topic, max_results=4)
    # source is "arxiv" or "mock"
"""
from __future__ import annotations

import logging
import textwrap

import arxiv

logger = logging.getLogger(__name__)

_client = arxiv.Client()


def fetch_from_arxiv(topic: str, max_results: int = 4) -> list[dict]:
    """
    Search arXiv for the topic and return a list of paper dicts
    matching the same schema as the mock corpus:
      {"title": str, "authors": str, "abstract": str}

    Raises RuntimeError if the search returns no results.
    """
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for result in _client.results(search):
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

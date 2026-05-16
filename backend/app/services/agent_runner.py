"""
AgentRunner — the research pipeline state machine.

Lifecycle:  PENDING → RETRIEVING → SYNTHESIZING → COMPLETED | FAILED

This module is the only place that:
  • mutates ResearchTask.status
  • calls NeMoClawClient
  • writes GeneratedHypothesis rows
  • appends AgentRunLog entries

FastAPI background tasks call run_research_pipeline() in a thread pool.
All DB operations use the passed session directly (no async ORM needed for SQLite).
"""
from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from app.db.models import ResearchTask, GeneratedHypothesis, AgentRunLog
from app.services.mock_data import get_papers, format_context_block
from app.services.nemoclaw_client import get_nemoclaw_client, NeMoClawError
from app.prompts.research_synthesis import SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_USER_TEMPLATE
from app.schemas.pydantic import HypothesisOutput

logger = logging.getLogger(__name__)


def _log(db: Session, task_id: str, stage: str, message: str) -> None:
    """Append an audit entry and flush without committing the outer transaction."""
    entry = AgentRunLog(task_id=task_id, stage=stage, message=message)
    db.add(entry)
    db.flush()


def _set_status(db: Session, task: ResearchTask, status: str, message: str = "") -> None:
    task.status = status
    task.updated_at = datetime.utcnow()
    _log(db, task.id, status, message or f"Transitioned to {status}")
    db.commit()
    logger.info("Task %s → %s", task.id, status)


def run_research_pipeline(task_id: str, topic: str, db: Session) -> None:
    """
    Entry point for FastAPI BackgroundTasks.

    All exceptions are caught at the top level to guarantee the task lands
    in FAILED state rather than hanging in SYNTHESIZING forever.
    """
    task: ResearchTask | None = db.query(ResearchTask).filter(ResearchTask.id == task_id).first()
    if not task:
        logger.error("Task %s not found — aborting pipeline.", task_id)
        return

    try:
        _execute_pipeline(task, topic, db)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed for task %s", task_id)
        task.error_message = str(exc)[:1000]
        _set_status(db, task, "FAILED", f"Unhandled error: {exc}")


def _execute_pipeline(task: ResearchTask, topic: str, db: Session) -> None:
    # ── Stage 1: Retrieve context ─────────────────────────────────────────────
    _set_status(db, task, "RETRIEVING", f"Fetching literature context for topic: '{topic}'")

    papers, source = get_papers(topic)
    context_block = format_context_block(papers)
    _log(
        db, task.id, "RETRIEVING",
        f"Retrieved {len(papers)} papers for topic '{topic}' (source: {source}).",
    )
    db.commit()

    # ── Stage 2: NeMoClaw synthesis ───────────────────────────────────────────
    _set_status(
        db, task, "SYNTHESIZING",
        "Submitting context to NeMoClaw for gap analysis and hypothesis generation.",
    )

    user_prompt = SYNTHESIS_USER_TEMPLATE.format(
        topic=topic,
        context_block=context_block,
    )

    client = get_nemoclaw_client()
    _log(
        db, task.id, "SYNTHESIZING",
        f"NeMoClaw client initialised ({type(client).__name__}). "
        f"Model: {client._model}. Max tokens: {client._max_tokens}.",
    )
    db.commit()

    try:
        structured: HypothesisOutput
        raw_json: str
        structured, raw_json = client.run(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=HypothesisOutput,
        )
    except NeMoClawError as exc:
        raise RuntimeError(f"NeMoClaw synthesis failed: {exc}") from exc

    _log(
        db, task.id, "SYNTHESIZING",
        f"NeMoClaw returned valid structured output. Confidence: {structured.confidence_score}.",
    )
    db.commit()

    # ── Stage 3: Persist structured output ───────────────────────────────────
    hypothesis = GeneratedHypothesis(
        task_id=task.id,
        gap_identified=structured.gap_identified,
        proposed_architecture=structured.proposed_architecture,
        evaluation_metric=structured.evaluation_metric,
        confidence_score=structured.confidence_score,
        raw_llm_output=raw_json,
    )
    db.add(hypothesis)

    _set_status(
        db, task, "COMPLETED",
        "Hypothesis persisted successfully. Pipeline complete.",
    )
    # Final commit handles both the hypothesis row and the COMPLETED status.
    db.commit()

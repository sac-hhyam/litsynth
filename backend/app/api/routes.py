"""
FastAPI router — all /api/v1/research endpoints.

Endpoint contract:
  POST   /research/analyze      → create task, kick off background pipeline
  GET    /research/task/{id}    → poll status (the state-machine demo endpoint)
  GET    /research/task/{id}/results → fetch final structured hypothesis
  GET    /research/task/{id}/logs    → audit trail for the pipeline run
  GET    /research/tasks        → list all tasks (ordered newest first)
  GET    /research/topics       → available demo topics
  DELETE /research/task/{id}    → remove a task and its output (cleanup)
"""
from __future__ import annotations

import uuid
import logging
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ResearchTask
from app.schemas.pydantic import (
    AnalyzeTopicRequest,
    TaskCreatedResponse,
    TaskStatusResponse,
    TaskResultResponse,
    HypothesisResponse,
    TaskLogsResponse,
    AgentLogEntry,
    TopicListResponse,
)
from app.services.agent_runner import run_research_pipeline
from app.services.mock_data import get_available_topics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["Research Agent"])


# ── POST /research/analyze ────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=TaskCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a research topic for autonomous analysis",
    description=(
        "Triggers the full Lit-Agent pipeline asynchronously. "
        "Returns a task_id immediately; poll `/task/{task_id}` to track progress."
    ),
)
def analyze_topic(
    body: AnalyzeTopicRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TaskCreatedResponse:
    task_id = str(uuid.uuid4())
    now = datetime.utcnow()

    task = ResearchTask(id=task_id, topic=body.topic, status="PENDING", created_at=now, updated_at=now)
    db.add(task)
    db.commit()
    db.refresh(task)

    logger.info("Created task %s for topic '%s'", task_id, body.topic)

    # Detach a new DB session for the background thread to avoid sharing
    # the request-scoped session across thread boundaries.
    from app.db.database import SessionLocal  # local import avoids circular

    def _run_in_background() -> None:
        bg_db = SessionLocal()
        try:
            run_research_pipeline(task_id=task_id, topic=body.topic, db=bg_db)
        finally:
            bg_db.close()

    background_tasks.add_task(_run_in_background)

    return TaskCreatedResponse(
        task_id=task_id,
        status="PENDING",
        message=(
            f"Pipeline started. Poll GET /api/v1/research/task/{task_id} "
            "to track status (PENDING → RETRIEVING → SYNTHESIZING → COMPLETED)."
        ),
    )


# ── GET /research/task/{task_id} ──────────────────────────────────────────────

@router.get(
    "/task/{task_id}",
    response_model=TaskStatusResponse,
    summary="Poll task status",
    description="Returns the current state-machine status for a given task.",
)
def get_task_status(
    task_id: str,
    db: Session = Depends(get_db),
) -> TaskStatusResponse:
    task = _get_task_or_404(task_id, db)
    return TaskStatusResponse(
        task_id=task.id,
        topic=task.topic,
        status=task.status,
        created_at=task.created_at,
        updated_at=task.updated_at,
        error_message=task.error_message,
    )


# ── GET /research/task/{task_id}/results ─────────────────────────────────────

@router.get(
    "/task/{task_id}/results",
    response_model=TaskResultResponse,
    summary="Fetch autonomous hypothesis output",
    description=(
        "Returns the structured NeMoClaw-generated hypothesis once the task "
        "reaches COMPLETED status. Returns 404 if the hypothesis has not been "
        "generated yet."
    ),
)
def get_task_results(
    task_id: str,
    db: Session = Depends(get_db),
) -> TaskResultResponse:
    task = _get_task_or_404(task_id, db)

    if task.status == "FAILED":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "status": "FAILED",
                "error": task.error_message or "Pipeline failed with no error message.",
            },
        )

    hypothesis_response = None
    if task.hypothesis:
        h = task.hypothesis
        hypothesis_response = HypothesisResponse(
            task_id=task.id,
            gap_identified=h.gap_identified,
            proposed_architecture=h.proposed_architecture,
            evaluation_metric=h.evaluation_metric,
            confidence_score=h.confidence_score,
            raw_llm_output=h.raw_llm_output,
            created_at=h.created_at,
        )

    return TaskResultResponse(
        task_id=task.id,
        topic=task.topic,
        status=task.status,
        hypothesis=hypothesis_response,
    )


# ── GET /research/task/{task_id}/logs ─────────────────────────────────────────

@router.get(
    "/task/{task_id}/logs",
    response_model=TaskLogsResponse,
    summary="Retrieve agent run audit trail",
    description="Returns every state-transition log entry for the given task.",
)
def get_task_logs(
    task_id: str,
    db: Session = Depends(get_db),
) -> TaskLogsResponse:
    task = _get_task_or_404(task_id, db)
    logs = [
        AgentLogEntry(stage=log.stage, message=log.message, timestamp=log.timestamp)
        for log in sorted(task.logs, key=lambda l: l.timestamp)
    ]
    return TaskLogsResponse(task_id=task_id, logs=logs)


# ── GET /research/tasks ───────────────────────────────────────────────────────

@router.get(
    "/tasks",
    response_model=list[TaskStatusResponse],
    summary="List all research tasks",
    description="Returns all tasks ordered by creation time (newest first).",
)
def list_tasks(
    db: Session = Depends(get_db),
) -> list[TaskStatusResponse]:
    tasks = db.query(ResearchTask).order_by(ResearchTask.created_at.desc()).all()
    return [
        TaskStatusResponse(
            task_id=t.id,
            topic=t.topic,
            status=t.status,
            created_at=t.created_at,
            updated_at=t.updated_at,
            error_message=t.error_message,
        )
        for t in tasks
    ]


# ── GET /research/topics ──────────────────────────────────────────────────────

@router.get(
    "/topics",
    response_model=TopicListResponse,
    summary="List available demo topics",
    description="Returns the pre-loaded literature corpus topics for demo use.",
)
def list_topics() -> TopicListResponse:
    topics = get_available_topics()
    return TopicListResponse(topics=topics, count=len(topics))


# ── DELETE /research/task/{task_id} ──────────────────────────────────────────

@router.delete(
    "/task/{task_id}",
    summary="Delete a task and its hypothesis",
    description="Hard-deletes the task row and cascades to hypothesis and logs.",
)
def delete_task(
    task_id: str,
    db: Session = Depends(get_db),
) -> Response:
    task = _get_task_or_404(task_id, db)
    if task.status in ("RETRIEVING", "SYNTHESIZING"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete a task that is currently running.",
        )
    db.delete(task)
    db.commit()
    return Response(status_code=204)


# ── Health check (module-level, no prefix) ───────────────────────────────────

health_router = APIRouter(tags=["Health"])


@health_router.get("/health", summary="Service health check")
def health_check() -> dict:
    return {"status": "ok", "service": "lit-agent"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_task_or_404(task_id: str, db: Session) -> ResearchTask:
    task = db.query(ResearchTask).filter(ResearchTask.id == task_id).first()
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )
    return task

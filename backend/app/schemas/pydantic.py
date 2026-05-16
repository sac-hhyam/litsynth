"""
Pydantic v2 schemas — the contract between HTTP, agent runtime, and the DB.

NeMoClaw uses HypothesisOutput as its response_model to enforce structured JSON.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Inbound ──────────────────────────────────────────────────────────────────

class AnalyzeTopicRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=5,
        max_length=300,
        examples=["Efficient LLM routing strategies for multi-task learning"],
    )


# ── NeMoClaw structured output contract ──────────────────────────────────────

class HypothesisOutput(BaseModel):
    """
    The schema NeMoClaw enforces on the LLM response.
    Every field maps 1:1 to a column in generated_hypotheses.
    """
    gap_identified: str = Field(
        description="The specific research gap or limitation found across the surveyed papers."
    )
    proposed_architecture: str = Field(
        description=(
            "A concrete novel architecture or method that addresses the identified gap. "
            "Name components, data flow, and training paradigm."
        )
    )
    evaluation_metric: str = Field(
        description=(
            "Primary metric and benchmark suite to validate the proposed architecture. "
            "Format: '<metric> on <benchmark>'."
        )
    )
    confidence_score: str = Field(
        description="Agent confidence in the hypothesis on a scale of LOW / MEDIUM / HIGH.",
        pattern="^(LOW|MEDIUM|HIGH)$",
    )


# ── Outbound ─────────────────────────────────────────────────────────────────

class TaskCreatedResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    topic: str
    status: str
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None


class HypothesisResponse(BaseModel):
    task_id: str
    gap_identified: str
    proposed_architecture: str
    evaluation_metric: str
    confidence_score: Optional[str]
    raw_llm_output: Optional[str] = None
    created_at: datetime


class TaskResultResponse(BaseModel):
    task_id: str
    topic: str
    status: str
    hypothesis: Optional[HypothesisResponse] = None


class AgentLogEntry(BaseModel):
    stage: str
    message: str
    timestamp: datetime


class TaskLogsResponse(BaseModel):
    task_id: str
    logs: list[AgentLogEntry]


class TopicListResponse(BaseModel):
    """Available mock topics for demo discoverability."""
    topics: list[str]
    count: int

"""
SQLAlchemy ORM models.

research_tasks  — tracks lifecycle of each research pipeline run
generated_hypotheses — persists the structured output produced by NeMoClaw
agent_run_logs  — append-only audit trail; every state transition is recorded
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer
from sqlalchemy.orm import relationship

from app.db.database import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


class ResearchTask(Base):
    __tablename__ = "research_tasks"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    topic = Column(String(512), nullable=False)
    # PENDING → RETRIEVING → SYNTHESIZING → COMPLETED | FAILED
    status = Column(String(32), nullable=False, default="PENDING")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    hypothesis = relationship(
        "GeneratedHypothesis", back_populates="task", uselist=False, cascade="all, delete-orphan"
    )
    logs = relationship(
        "AgentRunLog", back_populates="task", cascade="all, delete-orphan"
    )


class GeneratedHypothesis(Base):
    __tablename__ = "generated_hypotheses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(36), ForeignKey("research_tasks.id"), nullable=False, unique=True)

    # ── Core structured output fields (enforced by NeMoClaw) ──────────────────
    gap_identified = Column(Text, nullable=False)
    proposed_architecture = Column(Text, nullable=False)
    evaluation_metric = Column(String(512), nullable=False)
    # NeMoClaw returns confidence score alongside structured output
    confidence_score = Column(String(8), nullable=True)
    # Raw JSON blob from the LLM for full transparency / audit
    raw_llm_output = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    task = relationship("ResearchTask", back_populates="hypothesis")


class AgentRunLog(Base):
    __tablename__ = "agent_run_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(36), ForeignKey("research_tasks.id"), nullable=False)
    stage = Column(String(64), nullable=False)   # e.g. "RETRIEVING", "SYNTHESIZING"
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    task = relationship("ResearchTask", back_populates="logs")

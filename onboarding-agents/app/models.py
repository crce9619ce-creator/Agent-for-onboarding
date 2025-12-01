from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from .db import Base
import enum

class TaskStatusEnum(str, enum.Enum):
    on_track = "on_track"
    delayed = "delayed"
    at_risk = "at_risk"

class Run(Base):
    __tablename__ = "runs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    input_plan = Column(Text)
    logs = Column(JSON, nullable=True)
    tasks = relationship("Task", back_populates="run", cascade="all, delete-orphan")
    status_analyses = relationship("StatusAnalysis", back_populates="run", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="run", cascade="all, delete-orphan")

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text)
    due_date = Column(String, nullable=True)
    owner = Column(String, nullable=True)
    raw = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    run = relationship("Run", back_populates="tasks")

class StatusAnalysis(Base):
    __tablename__ = "status_analysis"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    task_id = Column(Integer, nullable=False)
    status = Column(Enum(TaskStatusEnum), nullable=False)
    explanation = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    run = relationship("Run", back_populates="status_analyses")

class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    note = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    run = relationship("Run", back_populates="recommendations")

from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from enum import Enum

class TaskSchema(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: Optional[str] = None
    owner: Optional[str] = None

class TaskExtractionOutput(BaseModel):
    tasks: List[TaskSchema]

class StatusEnum(str, Enum):
    on_track = "on_track"
    delayed = "delayed"
    at_risk = "at_risk"

class TaskStatus(BaseModel):
    task_title: str
    status: StatusEnum
    explanation: str

class StatusAnalysisOutput(BaseModel):
    analyses: List[TaskStatus]

class RecommendationSchema(BaseModel):
    notes: List[str]

class RunResponse(BaseModel):
    run_id: UUID
    timestamps: dict
    outputs: dict
    logs: Optional[dict] = None

import uuid
from datetime import datetime
from typing import List
from .schemas import TaskSchema, TaskExtractionOutput, StatusAnalysisOutput, RecommendationSchema, TaskStatus
from tenacity import retry, stop_after_attempt, wait_fixed

def call_llm(prompt: str) -> str:
    if "Kickoff" in prompt:
        return """
- title: Kickoff Call
  description: Schedule a 60-minute kickoff with stakeholders
  due_date: 2025-01-15
  owner: Priya
- title: Data Migration Prep
  description: Client to send customer data export
  due_date: 2025-01-20
  owner: (unassigned)
"""
    return "[]"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def task_extraction_agent(onboarding_text: str) -> TaskExtractionOutput:
    raw = call_llm(f"Extract tasks from: {onboarding_text}")
    tasks: List[TaskSchema] = []
    for block in raw.strip().split("\n- "):
        if not block.strip():
            continue
        lines = [l.strip() for l in block.split("\n") if ":" in l]
        data = {}
        for l in lines:
            k,v = l.split(":",1)
            data[k.strip().lower()] = v.strip()
        tasks.append(TaskSchema(
            title=data.get("title") or "Untitled",
            description=data.get("description"),
            due_date=data.get("due_date"),
            owner=data.get("owner")
        ))
    return TaskExtractionOutput(tasks=tasks)

@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
def status_analysis_agent(tasks: List[TaskSchema]) -> StatusAnalysisOutput:
    analyses = []
    now = datetime.utcnow()
    for t in tasks:
        if t.due_date is None or "(unassigned)" in (t.owner or "").lower():
            status = "at_risk"
            explanation = "Missing due date or owner."
        else:
            try:
                due = datetime.fromisoformat(t.due_date)
                if due < now:
                    status = "delayed"
                    explanation = "Due date is in the past."
                else:
                    status = "on_track"
                    explanation = "Valid owner & due date."
            except Exception:
                status = "at_risk"
                explanation = "Invalid due_date format."
        analyses.append(TaskStatus(task_title=t.title, status=status, explanation=explanation))
    return StatusAnalysisOutput(analyses=analyses)

@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
def recommendation_agent(status_analysis: StatusAnalysisOutput) -> RecommendationSchema:
    notes = []
    for a in status_analysis.analyses:
        if a.status == "delayed":
            notes.append(f"Escalate {a.task_title}: delayed.")
        elif a.status == "at_risk":
            notes.append(f"Investigate {a.task_title}: {a.explanation}")
    if not notes:
        notes.append("All tasks appear on track.")
    return RecommendationSchema(notes=notes)

def run_onboarding_pipeline(onboarding_text: str):
    timestamps = {}
    run_id = uuid.uuid4()
    timestamps['started_at'] = datetime.utcnow().isoformat()
    tasks_out = task_extraction_agent(onboarding_text)
    timestamps['task_extraction_finished'] = datetime.utcnow().isoformat()
    status_out = status_analysis_agent(tasks_out.tasks)
    timestamps['status_analysis_finished'] = datetime.utcnow().isoformat()
    rec_out = recommendation_agent(status_out)
    timestamps['recommendation_finished'] = datetime.utcnow().isoformat()
    return {
        "run_id": run_id,
        "timestamps": timestamps,
        "tasks_out": tasks_out,
        "status_out": status_out,
        "rec_out": rec_out
    }

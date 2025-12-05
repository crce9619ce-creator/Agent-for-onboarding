import os
import uuid
import json
import re
from datetime import datetime
from typing import List
from .schemas import TaskSchema, TaskExtractionOutput, StatusAnalysisOutput, RecommendationSchema, TaskStatus
from tenacity import retry, stop_after_attempt, wait_fixed
from .config import settings

# Optional: uses the official OpenAI Python package
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

def _extract_json(text: str) -> str:
    """
    Extract the first JSON object/array from model output.
    """
    # Attempt to find a JSON block
    json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if json_match:
        return json_match.group(1)
    # fallback: return whole text
    return text

def call_llm(prompt: str, max_tokens: int = 800) -> str:
    """
    Calls OpenAI ChatCompletion (synchronously) to produce a JSON payload.
    Expects OPENAI_API_KEY to be set in app.config.settings.OPENAI_API_KEY or in env.
    Returns the text response (string).
    """
    # If openai package isn't available, raise an informative error
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package is not installed. Add 'openai' to requirements and install it.")

    api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY in environment or .env.")

    openai.api_key = api_key

    system_msg = (
        "You are an assistant that extracts structured onboarding tasks from a freeform onboarding plan. "
        "Return valid JSON only (no explanatory text) with the following schema:\n\n"
        "{\n"
        '  \"tasks\": [\n'
        "    {\n"
        '      \"title\": string, \n'
        '      \"description\": string or null,\n'
        '      \"due_date\": string in ISO-8601 or null,\n'
        '      \"owner\": string or null\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "If a field is unknown, set it to null. Keep dates in YYYY-MM-DD or full ISO format if possible."
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system", "content": system_msg},
                {"role":"user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        # Get content
        content = resp["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        # raise so tenacity can retry
        raise

# 1) Task Extraction Agent (with Pydantic validation)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def task_extraction_agent(onboarding_text: str) -> TaskExtractionOutput:
    prompt = f"Extract tasks from the following onboarding plan. Provide only JSON matching the schema.\n\n{onboarding_text}"
    raw = call_llm(prompt)
    json_text = _extract_json(raw)
    try:
        parsed = json.loads(json_text)
    except Exception as e:
        # If parsing fails, raise to trigger retry
        raise ValueError(f"Failed to parse JSON from LLM response: {e}\\nResponse was: {raw}")

    tasks_data = parsed.get("tasks", [])
    tasks: List[TaskSchema] = []
    for t in tasks_data:
        tasks.append(TaskSchema(
            title=t.get("title") or "Untitled",
            description=t.get("description"),
            due_date=t.get("due_date"),
            owner=t.get("owner")
        ))
    return TaskExtractionOutput(tasks=tasks)

# 2) Status Analysis Agent (same rule-based logic as before)
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

# 3) Recommendation Agent
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

# Orchestration function
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

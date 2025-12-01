from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .agents import run_onboarding_pipeline
from .crud import persist_run_and_outputs
from .schemas import RunResponse
app = FastAPI(title="Onboarding Agents")

class InputPlan(BaseModel):
    onboarding_plan: str

@app.post("/run_onboarding_agents", response_model=RunResponse)
def run_onboarding_agents(payload: InputPlan):
    try:
        res = run_onboarding_pipeline(payload.onboarding_plan)
        persist_run_and_outputs(
            res["run_id"],
            payload.onboarding_plan,
            res["tasks_out"],
            res["status_out"],
            res["rec_out"],
            logs={"info":"pipeline run"}
        )
        return {
            "run_id": res["run_id"],
            "timestamps": res["timestamps"],
            "outputs": {
                "tasks": [t.dict() for t in res["tasks_out"].tasks],
                "status_analysis": [a.dict() for a in res["status_out"].analyses],
                "recommendations": res["rec_out"].notes
            },
            "logs": {"note":"stored successfully"}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

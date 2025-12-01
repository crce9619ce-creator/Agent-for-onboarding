# Onboarding Agents — multi-agent workflow

Implements a 3-step LangGraph-style workflow:
1. Task Extraction Agent (Pydantic-validated)
2. Status Analysis Agent
3. Recommendation Agent

Exposes:

POST /run_onboarding_agents
Input:
{
  "onboarding_plan": "..."
}

Output:
{
  "run_id": "<uuid>",
  "timestamps": {...},
  "outputs": {
    "tasks": [...],
    "status_analysis": [...],
    "recommendations": [...]
  },
  "logs": {...}
}

Setup (local):
1. Create virtual env and install:
   pip install -r requirements.txt
2. Create PostgreSQL DB and update DATABASE_URL in .env or app/config.py
3. Run SQL in migrations/create_tables.sql
4. Start server:
   uvicorn app.main:app --reload --port 8000

Notes:
- The `agents.py` currently contains a simple LLM stub and rule-based analyzers for demonstration. Replace `call_llm` with real LLM calls that return JSON matching Pydantic schemas.
- `tenacity` provides retry; you can customize retry policy or use LangGraph's built-in mechanisms when available.
- Tests:
   pytest app/tests -q

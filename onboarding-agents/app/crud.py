from .models import Run, Task, StatusAnalysis, Recommendation
from .db import SessionLocal
from contextlib import contextmanager

@contextmanager
def transactional_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def persist_run_and_outputs(run_id, input_plan, tasks_out, status_out, rec_out, logs=None):
    with transactional_session() as db:
        new_run = Run(id=run_id, input_plan=input_plan, logs={"logs": logs} if logs else {})
        db.add(new_run)
        db.flush()
        for t in tasks_out.tasks:
            db.add(Task(
                run_id=new_run.id,
                title=t.title,
                description=t.description,
                due_date=t.due_date,
                owner=t.owner,
                raw=t.dict()
            ))
        db.flush()
        for a in status_out.analyses:
            db.add(StatusAnalysis(
                run_id=new_run.id,
                task_id=0,
                status=a.status,
                explanation=a.explanation
            ))
        for note in rec_out.notes:
            db.add(Recommendation(run_id=new_run.id, note=note))
        db.flush()

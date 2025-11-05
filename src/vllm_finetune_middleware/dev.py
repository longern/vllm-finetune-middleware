import asyncio
import uuid

from fastapi import APIRouter, FastAPI, Body

from .middlewares import FineTuningMiddleware
from .worker import handler

router = APIRouter(prefix="/runpod", tags=["runpod"])

JOBS = {}
JOB_TASKS: dict[str, asyncio.Task] = {}


@router.post("/run")
async def create_job(input: dict = Body(...)):
    job_id = str(uuid.uuid4())
    job = {"id": job_id, "status": "IN_QUEUE"}
    JOBS[job_id] = job

    event = {"id": job_id, "input": input}

    coro = (
        handler(event)
        if asyncio.iscoroutinefunction(handler)
        else asyncio.to_thread(handler, event)
    )

    task = asyncio.create_task(coro)
    JOB_TASKS[job_id] = task

    return job


@router.get("/status/{job_id}")
async def retrieve_job(job_id: str):
    if job_id not in JOBS:
        return {"error": "Job not found"}, 404

    return JOBS.get(job_id)


@router.post("/cancel/{job_id}")
def cancel_job(job_id: str):
    if job_id not in JOBS:
        return {"error": "Job not found"}, 404

    task = JOB_TASKS.get(job_id)
    task.cancel()
    JOBS[job_id]["status"] = "CANCELLED"

    return JOBS[job_id]


app = FastAPI()
app.include_router(router)
app.add_middleware(FineTuningMiddleware)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)

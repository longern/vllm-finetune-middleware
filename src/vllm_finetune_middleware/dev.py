import asyncio
import json
import logging
import time
import traceback
import uuid
from typing import Any

from fastapi import APIRouter, Body, FastAPI

from .middlewares import FineTuningMiddleware
from .worker import handler

router = APIRouter(tags=["runpod"])

JOBS = {}
JOB_TASKS: dict[str, asyncio.Task] = {}


def task_done_callback_wrapper(job_id: str, start_time: float = time.perf_counter()):
    def wrapper(task: asyncio.Task):
        exception = task.exception()
        if task.cancelled():
            JOBS[job_id]["status"] = "CANCELLED"
        elif exception is not None:
            JOBS[job_id]["status"] = "FAILED"
            JOBS[job_id]["error"] = json.dumps(
                {
                    "error_message": str(exception),
                    "error_traceback": "".join(traceback.format_exception(exception)),
                    "error_type": str(type(exception)),
                }
            )
            logging.exception(f"Job {job_id} failed with exception", exc_info=exception)
        else:
            JOBS[job_id]["output"] = task.result()
            JOBS[job_id]["status"] = "COMPLETED"
            execution_time = int((time.perf_counter() - start_time) * 1000)
            JOBS[job_id]["executionTime"] = execution_time

        JOB_TASKS.pop(job_id, None)

    return wrapper


JOB_QUEUE_LOCK = asyncio.Lock()


async def queue_task(job_id: str, coro):
    async with JOB_QUEUE_LOCK:
        JOBS[job_id]["status"] = "RUNNING"
        await coro


@router.post("/run")
async def create_job(body: Any = Body(...)):
    job_id = str(uuid.uuid4())
    job = {"id": job_id, "status": "IN_QUEUE"}
    JOBS[job_id] = job

    event = {"id": job_id, "input": body["input"]}

    coro = (
        handler(event)
        if asyncio.iscoroutinefunction(handler)
        else asyncio.to_thread(handler, event)
    )

    task = asyncio.create_task(queue_task(job_id, coro))
    task.add_done_callback(task_done_callback_wrapper(job_id))
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

    if JOBS[job_id]["status"] not in ["IN_QUEUE", "RUNNING"]:
        return {"error": "Job cannot be cancelled"}, 400

    task = JOB_TASKS.get(job_id)
    task.cancel()

    return JOBS[job_id]


app = FastAPI()
app.include_router(router)
app.add_middleware(FineTuningMiddleware)

"""Router for fine-tuning jobs."""

import asyncio
import os
import tempfile
import time
from typing import Literal
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .files import get_s3_client

RUNPOD_ENDPOINT_URL = os.environ.get(
    "RUNPOD_ENDPOINT_URL", "http://localhost:8000/runpod"
)
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")

router = APIRouter(prefix="/fine_tuning", tags=["fine_tuning"])


class Job(BaseModel):
    model: str
    training_file: str
    method: dict | None = None
    suffix: str | None = None


class JobRead(Job):
    object: Literal["fine_tuning.job"] = "fine_tuning.job"
    id: str
    status: str
    created_at: int
    finished_at: int | None = None
    fine_tuned_model: str | None = None
    error: dict | None = None
    result_files: list[str] | None = None


JOBS: dict[str, JobRead] = {}

STATUS_MAP = {
    "IN_QUEUE": "queued",
    "COMPLETED": "succeeded",
    "RUNNING": "running",
    "CANCELLED": "cancelled",
    "FAILED": "failed",
    "TIMED_OUT": "failed",
}


async def job_daemon(job_id: str):
    while True:
        await asyncio.sleep(5)
        job = await retrieve_job(job_id)
        if job.status in ("failed", "cancelled"):
            return

        if job.status == "succeeded":
            break

    # Download model artifacts
    s3_client = get_s3_client()
    s3_model_hub_url = os.getenv("AWS_MODEL_HUB_URL")
    if s3_model_hub_url is None:
        return

    parsed_url = urlparse(s3_model_hub_url)
    bucket_name = parsed_url.netloc
    path = parsed_url.path.lstrip("/")
    model_directory_key = os.path.join(path, f"{job_id}")

    job = JOBS[job_id]
    prefix = f"ft.{job.suffix}" if job.suffix else "ft."

    with tempfile.TemporaryDirectory(prefix=prefix) as tmpdir:
        # Download model directory
        for obj in s3_client.list_objects_v2(
            Bucket=bucket_name, Prefix=model_directory_key
        ).get("Contents", []):
            file_key = obj["Key"]
            relative_path = os.path.relpath(file_key, model_directory_key)
            local_path = os.path.join(tmpdir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3_client.download_file(bucket_name, file_key, local_path)

        model_name = os.path.basename(tmpdir).replace(".", ":")

        async with httpx.AsyncClient() as client:
            await client.post(
                "http://localhost:8000//v1/load_lora_adapter",
                json={"lora_name": model_name, "lora_path": tmpdir},
            )


@router.post("/jobs", response_model=JobRead)
async def create_job(job: Job):
    client = httpx.AsyncClient()
    resp = await client.post(
        RUNPOD_ENDPOINT_URL + "/run",
        json={"input": job.model_dump()},
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
        },
    )

    if resp.is_error:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    body = resp.json()

    asyncio.create_task(job_daemon(body["id"]))

    job_read = JobRead(
        **job.model_dump(),
        id=body["id"],
        status="queued",
        created_at=int(time.time()),
        finished_at=None,
        fine_tuned_model=None,
        error=None,
        result_files=None,
    )
    JOBS[job_read.id] = job_read

    return job_read


@router.get("/jobs")
async def list_jobs():
    return {"data": list(reversed(JOBS.values()))}


@router.get("/jobs/{job_id}", response_model=JobRead)
async def retrieve_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            RUNPOD_ENDPOINT_URL + "/status/" + job_id,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
            },
        )

        if resp.status_code == 404:
            JOBS.pop(job_id, None)
            raise HTTPException(status_code=404, detail="Job not found")
        elif resp.is_error:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        body = resp.json()

    job_read = JOBS[job_id]
    job_read.status = STATUS_MAP[body["status"]]
    if job_read.status in ("succeeded", "failed", "cancelled"):
        job_read.finished_at = int(time.time())

    return job_read

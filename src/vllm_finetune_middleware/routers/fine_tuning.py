"""Router for fine-tuning jobs."""

import asyncio
import logging
import os
import tempfile
import time
from typing import Literal
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .files import get_s3_client

RUNPOD_ENDPOINT_URL = os.getenv("RUNPOD_ENDPOINT_URL", "http://localhost:8000/runpod")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")

router = APIRouter(prefix="/fine_tuning", tags=["fine_tuning"])


class Job(BaseModel):
    model: str
    training_file: str
    method: dict | None = None
    suffix: str | None = None


class JobError(BaseModel):
    code: str
    message: str
    param: str | None = None


class JobRead(Job):
    object: Literal["fine_tuning.job"] = "fine_tuning.job"
    id: str
    status: str
    created_at: int
    finished_at: int | None = None
    fine_tuned_model: str | None = None
    error: JobError | None = None
    result_files: list[str] | None = None


JOBS: dict[str, JobRead] = {}

STATUS_MAP = {
    "IN_QUEUE": "queued",
    "COMPLETED": "succeeded",
    "IN_PROGRESS": "running",
    "CANCELLED": "cancelled",
    "FAILED": "failed",
    "TIMED_OUT": "failed",
}


def download_s3_directory(s3_directory_url: str, tempdir: str):
    s3_client = get_s3_client()

    parsed_url = urlparse(s3_directory_url)
    bucket_name = parsed_url.netloc
    path = parsed_url.path.lstrip("/")
    model_directory_key = path if path.endswith("/") else path + "/"

    for obj in s3_client.list_objects_v2(
        Bucket=bucket_name, Prefix=model_directory_key
    ).get("Contents", []):
        file_key = obj["Key"]
        if file_key.endswith("/"):  # Skip directory entries
            continue

        relative_path = os.path.relpath(file_key, model_directory_key)
        local_path = os.path.join(tempdir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket_name, file_key, local_path)


async def job_daemon(job_id: str):
    while True:
        await asyncio.sleep(5)

        try:
            job = await retrieve_job(job_id)
        except HTTPException as exception:
            if exception.status_code == 404:
                logging.warning("Job %s not found, stopping daemon", job_id)
                JOBS.pop(job_id, None)
                return

        if job.status in ("failed", "cancelled"):
            return

        if job.status == "succeeded":
            break

    artifacts_url = os.getenv("AWS_ARTIFACTS_URL")
    if artifacts_url is None:
        return

    try:
        job = JOBS[job_id]
        prefix = f"ft.{job.suffix}." if job.suffix else "ft."

        model_download_dir = os.getenv("MODEL_DOWNLOAD_DIR")
        tempdir = tempfile.mkdtemp(prefix=prefix, dir=model_download_dir)

        model_s3_path = os.path.join(artifacts_url, job_id, "model")
        download_s3_directory(model_s3_path, tempdir)

        model_name = os.path.basename(tempdir).replace(".", ":")
        job.fine_tuned_model = model_name

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:8000/v1/load_lora_adapter",
                json={"lora_name": model_name, "lora_path": tempdir},
            )
            resp.raise_for_status()

    except Exception as e:
        logging.exception(f"Failed to download model artifacts for job {job_id}: {e}")


@router.post("/jobs", response_model=JobRead)
async def create_job(job: Job):
    client = httpx.AsyncClient()

    extra_args = {}
    if os.getenv("RUNPOD_WEBHOOK_URL"):
        extra_args["webhook"] = os.environ["RUNPOD_WEBHOOK_URL"]

    resp = await client.post(
        RUNPOD_ENDPOINT_URL + "/run",
        json={"input": job.model_dump(), **extra_args},
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

    if body.get("error"):
        job_read.error = JobError(code="error", message=body["error"])

    return job_read


@router.post("/jobs/{job_id}/cancel", response_model=JobRead)
async def cancel_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            RUNPOD_ENDPOINT_URL + "/cancel/" + job_id,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
            },
        )

        if resp.is_error:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return JOBS[job_id]

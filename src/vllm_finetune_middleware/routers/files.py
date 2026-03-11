"""Router for file upload and download."""

import json
import os
import shutil
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException, UploadFile
from starlette.responses import FileResponse, StreamingResponse

router = APIRouter(prefix="/files", tags=["files"])

WORKER_VOLUME_DIR = os.getenv("WORKER_VOLUME_DIR", os.path.expanduser("~/.lawftune"))


def get_s3_client():
    return boto3.client(
        service_name="s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.getenv("AWS_REGION"),
        endpoint_url=os.environ["AWS_S3_ENDPOINT"],
        config=Config(
            s3={"addressing_style": os.getenv("AWS_S3_ADDRESSING_STYLE", "auto")}
        ),
    )


def get_upload_url() -> str:
    return os.getenv("AWS_UPLOAD_URL", os.path.join(WORKER_VOLUME_DIR, "files"))


def is_s3_upload_url(upload_url: str) -> bool:
    return urlparse(upload_url).scheme.lower() == "s3"


def get_local_upload_dir(upload_url: str) -> Path:
    parsed_url = urlparse(upload_url)

    if parsed_url.scheme == "file":
        local_dir = Path(unquote(parsed_url.path)).expanduser()
    elif parsed_url.scheme == "":
        local_dir = Path(upload_url).expanduser()
    else:
        raise ValueError(
            "AWS_UPLOAD_URL must be an s3:// URL, a file:// URL, or a local path."
        )

    if not local_dir.is_absolute():
        local_dir = Path(WORKER_VOLUME_DIR) / local_dir

    return local_dir


def get_local_file_path(file_id: str) -> Path:
    return get_local_upload_dir(get_upload_url()) / file_id


@router.post("")
async def upload_file(file: UploadFile):
    upload_url = get_upload_url()
    file_id = str(uuid.uuid4())

    if is_s3_upload_url(upload_url):
        s3_client = get_s3_client()
        parsed_url = urlparse(upload_url)

        bucket_name = parsed_url.netloc
        path = parsed_url.path.lstrip("/")
        upload_key = os.path.join(path, file_id)

        s3_client.upload_fileobj(
            Fileobj=file.file,
            Bucket=bucket_name,
            Key=upload_key,
            ExtraArgs={
                "ContentType": file.content_type or "application/octet-stream",
                "Metadata": {"filename": file.filename or ""},
            },
        )
    else:
        file_path = get_local_file_path(file_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("wb") as dst:
            shutil.copyfileobj(file.file, dst)

    return {"object": "file", "id": file_id}


@router.get("/{file_id}/content")
async def download_file(file_id: str):
    upload_url = get_upload_url()

    if is_s3_upload_url(upload_url):
        s3_client = get_s3_client()
        parsed_url = urlparse(upload_url)

        bucket_name = parsed_url.netloc
        path = parsed_url.path.lstrip("/")
        download_key = os.path.join(path, file_id)

        try:
            s3_object = s3_client.get_object(Bucket=bucket_name, Key=download_key)

            content_type = s3_object.get("ContentType", "application/octet-stream")

            return StreamingResponse(
                s3_object["Body"].iter_chunks(chunk_size=8192), media_type=content_type
            )

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise HTTPException(status_code=404, detail="File not found")
            raise

    file_path = get_local_file_path(file_id)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path, media_type="application/octet-stream", filename=file_id
    )

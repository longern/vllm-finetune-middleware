"""Router for file upload and download."""

import os
import uuid
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException, UploadFile
from starlette.responses import StreamingResponse

router = APIRouter(prefix="/files", tags=["files"])

RUNPOD_ENDPOINT_URL = os.getenv("RUNPOD_ENDPOINT_URL", "http://localhost:8000/runpod")


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


@router.post("")
async def upload_file(file: UploadFile):
    s3_client = get_s3_client()

    s3_upload_url = os.environ["AWS_UPLOAD_URL"]
    parsed_url = urlparse(s3_upload_url)

    bucket_name = parsed_url.netloc
    path = parsed_url.path.lstrip("/")

    file_id = str(uuid.uuid4())
    upload_key = os.path.join(path, file_id)

    s3_client.upload_fileobj(
        Fileobj=file.file,
        Bucket=bucket_name,
        Key=upload_key,
        ExtraArgs={
            "ContentType": file.content_type or "application/octet-stream",
            "Metadata": {"filename": file.filename},
        },
    )

    return {"object": "file", "id": file_id}


@router.get("/{file_id}/content")
async def download_file(file_id: str):
    s3_client = get_s3_client()

    s3_upload_url = os.environ["AWS_UPLOAD_URL"]
    parsed_url = urlparse(s3_upload_url)

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
        else:
            raise

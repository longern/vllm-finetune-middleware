import asyncio
import os
import tempfile

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8000/v1/unload_lora_adapter",
            json={"lora_name": model_id},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return {"id": model_id, "deleted": True}


@router.post("/models/{model_id}/push_to_hub")
async def push_model_to_hub(model_id: str, repo_id: str, hf_token: str):
    from huggingface_hub import HfApi

    adapter_path = os.path.join(tempfile.gettempdir(), model_id.replace(":", "."))

    api = HfApi(token=hf_token)

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    future = api.upload_folder(
        folder_path=adapter_path,
        repo_id=repo_id,
        repo_type="model",
        run_as_future=True,
    )

    await asyncio.wrap_future(future)

    return JSONResponse({"id": model_id, "pushed": True})

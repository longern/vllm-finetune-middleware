"""RunPod middleware for routing fine-tuning requests to a local FastAPI app."""

import httpx
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .routers import files, fine_tuning


router = APIRouter(prefix="/v1")
router.include_router(files.router)
router.include_router(fine_tuning.router)


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


app = FastAPI()
app.include_router(router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

"""RunPod middleware for routing fine-tuning requests to a local FastAPI app."""

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .routers import files, fine_tuning, models

router = APIRouter(prefix="/v1")
router.include_router(files.router)
router.include_router(fine_tuning.router)
router.include_router(models.router)

app = FastAPI()
app.include_router(router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

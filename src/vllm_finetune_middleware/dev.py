from fastapi import FastAPI

from .internal_runpod import router
from .middlewares import FineTuningMiddleware


app = FastAPI()
app.include_router(router)
app.add_middleware(FineTuningMiddleware)

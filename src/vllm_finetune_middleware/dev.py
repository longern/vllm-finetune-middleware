import logging

from fastapi import FastAPI
from fastapi.responses import Response

from .internal_runpod import router
from .middlewares import FineTuningMiddleware


class MetricsLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.args is None or len(record.args) < 5:
            return True

        _, method, pathname, _, status_code, *_ = record.args

        if method == "GET" and pathname in {"/metrics", "/metrics/"} and status_code == 200:
            return False

        return True


uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(MetricsLogFilter())

app = FastAPI()
app.include_router(router)
app.add_middleware(FineTuningMiddleware)

try:
    import prometheus_client

    @app.get("/metrics")
    @app.get("/metrics/")
    async def metrics():
        return Response(
            content=prometheus_client.generate_latest(),
            media_type=prometheus_client.CONTENT_TYPE_LATEST,
        )

except ImportError:
    pass

"""RunPod middleware for routing fine-tuning requests to a local FastAPI app."""

from typing import List, Optional

from fastapi import FastAPI
from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match
from starlette.types import ASGIApp, Message

from .asgi import app


def resolve_route(
    app: FastAPI,
    path: str,
    method: Optional[str] = None,
):
    """
    Resolve a route in a FastAPI application based on the given path and method.
    """
    scope = {"type": "http", "path": path}
    if method:
        scope["method"] = method.upper()

    stack = [(app.router.routes, scope)]
    while stack:
        routes, sc = stack.pop()
        for route in routes:
            match, child = route.matches(sc)
            if match is Match.FULL:
                return route, (child or {})
            if match is Match.PARTIAL:
                subapp = getattr(route, "app", None)
                subroutes = getattr(getattr(subapp, "router", None), "routes", None)
                if subroutes is not None:
                    stack.append((subroutes, child))

    return None


class FineTuningMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
    ) -> None:
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path
        if resolve_route(app, path, request.method):
            return await self._forward_to_app(request)
        return await call_next(request)

    async def _forward_to_app(self, request: Request) -> Response:
        scope = request.scope.copy()
        scope["root_path"] = ""
        receive = request.receive

        send_queue: List[Message] = []
        response_started = False

        async def send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            send_queue.append(message)

        await app(scope, receive, send)

        assert response_started, "Response was not started"

        headers = MutableHeaders()
        body = b""
        status_code = 200

        for message in send_queue:
            if message["type"] == "http.response.start":
                status_code = message["status"]
                for name, value in message.get("headers", []):
                    headers.append(name.decode("latin-1"), value.decode("latin-1"))
            elif message["type"] == "http.response.body":
                body += message.get("body", b"")

        return Response(content=body, status_code=status_code, headers=dict(headers))

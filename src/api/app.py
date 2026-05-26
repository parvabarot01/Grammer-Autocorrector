"""FastAPI application entrypoint for the Grammar Autocorrector API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.api.runtime import APIRuntime
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    config = load_config()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        LOGGER.info("Starting Grammar Autocorrector API.")
        application.state.config = config
        application.state.runtime = APIRuntime(application.state.config)
        application.state.runtime.initialize()
        try:
            yield
        finally:
            LOGGER.info("Stopping Grammar Autocorrector API.")
            application.state.runtime.shutdown()

    app = FastAPI(
        title="Grammar Autocorrector API",
        description=(
            "NLP-based grammar correction using T5, BERT, and a RAG pipeline."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.config = config
    app.state.runtime = APIRuntime(app.state.config)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        request.state.request_id = str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = perf_counter()
        response = await call_next(request)
        processing_time_ms = (perf_counter() - started) * 1000
        request_id = getattr(request.state, "request_id", "unknown")
        LOGGER.info(
            "%s %s completed in %.2f ms request_id=%s",
            request.method,
            request.url.path,
            processing_time_ms,
            request_id,
        )
        response.headers["X-Process-Time-MS"] = f"{processing_time_ms:.3f}"
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(  # type: ignore[no-untyped-def]
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc.detail), "request_id": request_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(  # type: ignore[no-untyped-def]
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        LOGGER.exception("Unhandled API error request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "request_id": request_id},
        )

    app.include_router(router)
    return app


app = create_app()

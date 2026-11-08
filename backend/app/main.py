"""FastAPI application factory and lifecycle management."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .api.routes import router
from .services.search_pipeline import SearchPipeline, build_pipeline

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load persisted retrieval artifacts once when the server starts."""
    config.ensure_dirs()
    try:
        app.state.pipeline = build_pipeline()
        app.state.load_error = None
    except Exception as exc:  # Keep health endpoint available for diagnostics.
        app.state.pipeline = None
        app.state.load_error = str(exc)
    yield
    # FAISS/model objects are released with the application state.
    app.state.pipeline = None


def create_app(pipeline: SearchPipeline | None = None) -> FastAPI:
    app = FastAPI(
        title="CodeFinder API",
        description=("Developer error and solution search using lexical, semantic, "
                     "and hybrid information retrieval."),
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan if pipeline is None else None,
    )
    # The frontend is a public, unauthenticated SPA; no owner/auth checks are
    # intentionally present. Restrict browser origins rather than allowing
    # arbitrary cross-origin credentialed requests.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:4173", "http://127.0.0.1:4173",
        ],
        allow_credentials=False,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(router)
    if pipeline is not None:
        app.state.pipeline = pipeline
        app.state.load_error = None
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)

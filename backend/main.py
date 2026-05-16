"""
Application entry point.

Startup order:
  1. Configure logging
  2. Initialise SQLite schema (idempotent — safe to call on every restart)
  3. Mount routers
  4. Expose FastAPI app to uvicorn

Run locally:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Run on Brev (from project root):
  uvicorn main:app --host 0.0.0.0 --port 8000

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.db.database import init_db
from app.api.routes import router, health_router
from app.utils.logging import configure_logging

settings = get_settings()
configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=(
        "**LitSynth** is a backend research synthesis API powered by NVIDIA NIM (LLaMA 3.1 70B). "
        "Submit any research topic and the pipeline will:\n\n"
        "1. Retrieve relevant papers from arXiv\n"
        "2. Synthesise the core research gap across those papers\n"
        "3. Generate a structured, database-persisted experiment hypothesis\n\n"
        "All requests are async — use the `/task/{id}` polling endpoint to watch the "
        "pipeline move through its states in real time, then fetch results once `COMPLETED`."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(router, prefix=settings.API_PREFIX)

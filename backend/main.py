"""
Application entry point — DECOMMISSIONED (feat/nemoclaw-discord-agent)
=======================================================================

The REST/Swagger interface has been superseded by the NeMoClaw Discord bot.
The primary interface is now bot.py, which routes !synthesize commands through
the lit-synth-sandbox OpenShell sandbox via nemoclaw exec.

To run the new interface:
    cd backend
    source venv/bin/activate
    pip install discord.py
    export DISCORD_BOT_TOKEN=<your-token>
    python bot.py

To restore the FastAPI server (e.g. for local debugging), uncomment the
app.include_router lines at the bottom of this file and run:
    uvicorn main:app --host 0.0.0.0 --port 8000

Swagger UI was at: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.db.database import init_db
from app.utils.logging import configure_logging

# Route imports retained for reference — not mounted in this branch.
# from app.api.routes import router, health_router

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
        "⚠️ REST interface decommissioned on feat/nemoclaw-discord-agent branch.\n\n"
        "The active interface is the NeMoClaw Discord bot (`bot.py`). "
        "Use `!synthesize <topic>` in Discord to trigger synthesis."
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

# ── REST routes DISABLED — Discord bot is the active interface ────────────────
# Uncomment to restore the Swagger UI for debugging:
#
# from app.api.routes import router, health_router
# app.include_router(health_router)
# app.include_router(router, prefix=settings.API_PREFIX)

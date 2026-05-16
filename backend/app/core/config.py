"""
Application configuration — loads from .env via python-dotenv.
All secrets and tuneable constants live here; nothing else imports os.environ directly.
"""
import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── NVIDIA NIM / NeMoClaw ────────────────────────────────────────────────
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    # NIM inference base URL — swap for local NeMoClaw endpoint on Brev
    NIM_BASE_URL: str = os.getenv(
        "NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )
    # Model served via NIM on the Brev instance
    NIM_MODEL: str = os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct")

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./litsynth.db")

    # ── Agent tuning ─────────────────────────────────────────────────────────
    MAX_SYNTHESIS_TOKENS: int = int(os.getenv("MAX_SYNTHESIS_TOKENS", "1024"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.4"))

    # ── App metadata ─────────────────────────────────────────────────────────
    APP_TITLE: str = "LitSynth: Research Hypothesis Synthesiser"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()

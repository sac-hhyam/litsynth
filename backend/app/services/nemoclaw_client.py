"""
NeMoClaw Client — orchestration adapter between FastAPI and NVIDIA inference.

Client hierarchy:
  NeMoClawClient          — base: raw NIM API via httpx (always works)
    └── NeMoGuardrailsClient — preferred: NeMo Guardrails LLMRails over NIM
  MockNeMoClawClient      — offline fallback when no API key

Factory get_nemoclaw_client() selects automatically:
  NVIDIA_API_KEY set  → NeMoGuardrailsClient (with NIM fallback if rails fail)
  NVIDIA_API_KEY unset → MockNeMoClawClient

What NeMo Guardrails adds over raw NIM:
  • Colang flow enforcement (synthesiser.co defines agent behaviour)
  • Input/output guardrails (off-topic requests are rejected)
  • Structured dialog management via LLMRails runtime
  • Native NVIDIA agent orchestration layer
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import httpx
from pydantic import ValidationError

from app.core.config import get_settings
from app.schemas.pydantic import HypothesisOutput

logger = logging.getLogger(__name__)
settings = get_settings()

MAX_RETRIES = 2
RETRY_DELAY_S = 1.5

# Path to the NeMo Guardrails config directory
RAILS_CONFIG_PATH = Path(__file__).parent.parent.parent / "config"


class NeMoClawError(Exception):
    """Raised when the orchestration layer cannot produce a valid structured output."""


# ── Base client: raw NIM via httpx ───────────────────────────────────────────

class NeMoClawClient:
    """
    Base client — calls NVIDIA NIM /chat/completions directly via httpx.
    Used as the fallback when NeMo Guardrails is unavailable.
    """

    def __init__(self) -> None:
        self._base_url = settings.NIM_BASE_URL.rstrip("/")
        self._api_key = settings.NVIDIA_API_KEY
        self._model = settings.NIM_MODEL
        self._max_tokens = settings.MAX_SYNTHESIS_TOKENS
        self._temperature = settings.LLM_TEMPERATURE

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[HypothesisOutput],
    ) -> tuple[HypothesisOutput, str]:
        """
        Execute inference and return (structured_output, raw_json_string).
        Retries up to MAX_RETRIES times on malformed responses.
        """
        last_error: Optional[Exception] = None
        raw_text = ""

        for attempt in range(1, MAX_RETRIES + 2):
            try:
                raw_text = self._call_nim(system_prompt, user_prompt)
                logger.debug("Raw response (attempt %d): %s", attempt, raw_text[:300])
                structured = self._parse_and_validate(raw_text, response_model)
                logger.info(
                    "NeMoClaw inference succeeded on attempt %d (confidence=%s)",
                    attempt, structured.confidence_score,
                )
                return structured, raw_text

            except (NeMoClawError, ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "NeMoClaw attempt %d failed: %s — %s",
                    attempt, type(exc).__name__, str(exc)[:120],
                )
                if attempt <= MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S)

        raise NeMoClawError(
            f"Failed after {MAX_RETRIES + 1} attempts. "
            f"Last error: {last_error}. Raw snippet: {raw_text[:200]}"
        )

    def _call_nim(self, system_prompt: str, user_prompt: str) -> str:
        """POST to NIM /chat/completions and return the raw assistant message."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload, headers=headers,
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise NeMoClawError(
                f"NIM HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.RequestError as exc:
            raise NeMoClawError(f"NIM connection error: {exc}") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise NeMoClawError(f"Unexpected NIM response shape: {data}") from exc

        usage = data.get("usage", {})
        logger.info(
            "NIM token usage — prompt: %s  completion: %s  total: %s",
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
            usage.get("total_tokens", "?"),
        )
        return content

    @staticmethod
    def _parse_and_validate(raw: str, model: type[HypothesisOutput]) -> HypothesisOutput:
        """Strip markdown fences, parse JSON, validate against Pydantic schema."""
        fence_pattern = r"```(?:json)?\s*([\s\S]+?)\s*```"
        match = re.search(fence_pattern, raw)
        json_str = match.group(1) if match else raw.strip()
        brace_idx = json_str.find("{")
        if brace_idx > 0:
            json_str = json_str[brace_idx:]
        return model.model_validate(json.loads(json_str))


# ── NeMo Guardrails client: Colang-orchestrated inference ─────────────────────

class NeMoGuardrailsClient(NeMoClawClient):
    """
    NeMo Guardrails LLMRails client — the real NeMoClaw integration.

    Loads Colang flows from backend/config/ and uses LLMRails to orchestrate
    inference through the NIM endpoint with guardrails enforced.

    Falls back to raw NIM (parent _call_nim) if rails fail to load or error.

    What this adds over the base client:
      • synthesiser.co Colang flow governs agent behaviour
      • Off-topic inputs are rejected before hitting NIM
      • NVIDIA's agent runtime manages the dialog state
    """

    def __init__(self) -> None:
        super().__init__()
        self._rails = None
        self._rails_available = False
        self._load_rails()

    def _load_rails(self) -> None:
        """Attempt to load NeMo Guardrails from the config directory."""
        try:
            from nemoguardrails import RailsConfig, LLMRails  # type: ignore
            from langchain_openai import ChatOpenAI  # type: ignore

            if not RAILS_CONFIG_PATH.exists():
                logger.warning(
                    "NeMo Guardrails config dir not found at %s — using raw NIM.",
                    RAILS_CONFIG_PATH,
                )
                return

            # Build the LangChain LLM directly so the NIM API key is injected
            # at the Python level — bypasses unreliable ${VAR} YAML substitution.
            llm = ChatOpenAI(
                model=self._model,
                api_key=self._api_key,          # type: ignore[arg-type]
                base_url=self._base_url,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

            config = RailsConfig.from_path(str(RAILS_CONFIG_PATH))
            self._rails = LLMRails(config, llm=llm)
            self._rails_available = True
            logger.info(
                "NeMo Guardrails loaded from %s — Colang Synthesiser agent active "
                "(LLM: %s @ %s).",
                RAILS_CONFIG_PATH, self._model, self._base_url,
            )

        except ImportError as exc:
            logger.warning(
                "Required package missing (%s) — falling back to raw NIM. "
                "Install with: pip install nemoguardrails langchain-openai",
                exc,
            )
        except Exception as exc:
            logger.warning(
                "NeMo Guardrails failed to load (%s: %s) — falling back to raw NIM.",
                type(exc).__name__, exc,
            )

    def _call_nim(self, system_prompt: str, user_prompt: str) -> str:  # type: ignore[override]
        """
        Route through NeMo Guardrails LLMRails if available,
        otherwise fall back to the parent raw NIM call.
        """
        if not self._rails_available or self._rails is None:
            logger.info("NeMo Guardrails unavailable — using raw NIM fallback.")
            return super()._call_nim(system_prompt, user_prompt)

        return self._call_rails(system_prompt, user_prompt)

    def _call_rails(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call LLMRails.generate_async() synchronously via asyncio.
        NeMo Guardrails applies the Colang flow before/after NIM inference.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    self._rails.generate_async(messages=messages)
                )
            finally:
                loop.close()

            # LLMRails returns either a string or a dict with 'content'
            if isinstance(response, dict):
                content = response.get("content", str(response))
            else:
                content = str(response)

            logger.info(
                "NeMo Guardrails Synthesiser agent completed (rails=%s).",
                RAILS_CONFIG_PATH.name,
            )
            return content

        except Exception as exc:
            logger.warning(
                "NeMo Guardrails runtime error (%s: %s) — falling back to raw NIM.",
                type(exc).__name__, exc,
            )
            return super()._call_nim(system_prompt, user_prompt)


# ── Mock fallback ─────────────────────────────────────────────────────────────

class MockNeMoClawClient(NeMoClawClient):
    """
    Deterministic offline mock — no API key required.
    Returns realistic hypotheses for the 4 hardcoded topics.
    """

    _MOCK_RESPONSES: dict[str, dict] = {
        "efficient llm routing": {
            "gap_identified": (
                "Existing routing systems make per-token or static routing decisions "
                "that ignore inter-token semantic continuity and cannot adapt to "
                "distribution shift at inference time."
            ),
            "proposed_architecture": (
                "SeqRouter: a sequence-level routing transformer that encodes a sliding "
                "window of hidden states via a lightweight 2-layer cross-attention module. "
                "Updated online via EMA of accept-rate feedback from speculative verification."
            ),
            "evaluation_metric": "MMLU accuracy degradation vs. routing latency (ms) on MT-Bench",
            "confidence_score": "HIGH",
        },
        "vision transformer robustness": {
            "gap_identified": (
                "ViTs concentrate attention on a small subset of salient tokens with no "
                "dynamic mechanism to redistribute attention when those tokens are corrupted."
            ),
            "proposed_architecture": (
                "AdaptiveShield-ViT: uncertainty-aware attention re-routing layer that "
                "redistributes attention to globally consistent tokens via sparse softmax."
            ),
            "evaluation_metric": "mCE on ImageNet-C and ASR on Patch-Fool adversarial set",
            "confidence_score": "MEDIUM",
        },
        "llm hallucination detection": {
            "gap_identified": (
                "Multi-sample consistency methods fail on systematic hallucinations. "
                "Single-pass verifiers require curated knowledge bases."
            ),
            "proposed_architecture": (
                "ClaimGraph: decomposes text into atomic claims, builds a dependency graph, "
                "and scores each claim via a retrieval-augmented passage-trust classifier."
            ),
            "evaluation_metric": "F1 hallucination detection rate on FActScore benchmark",
            "confidence_score": "HIGH",
        },
        "protein structure prediction": {
            "gap_identified": (
                "Current models produce single static structures with poorly calibrated "
                "confidence for disordered regions and novel covalent binders."
            ),
            "proposed_architecture": (
                "EnsembleFlow: flow-matching generative model conditioned on ESMFold embeddings "
                "that samples conformational distributions rather than single structures."
            ),
            "evaluation_metric": "TM-score diversity on CATH conformational ensemble benchmark",
            "confidence_score": "MEDIUM",
        },
    }
    _DEFAULT_KEY = "efficient llm routing"

    def _call_nim(self, system_prompt: str, user_prompt: str) -> str:  # type: ignore[override]
        logger.info("MockNeMoClawClient: returning deterministic hypothesis (no API key set).")
        for key in self._MOCK_RESPONSES:
            if key in user_prompt.lower():
                return json.dumps(self._MOCK_RESPONSES[key])
        return json.dumps(self._MOCK_RESPONSES[self._DEFAULT_KEY])


# ── Factory ───────────────────────────────────────────────────────────────────

def get_nemoclaw_client() -> NeMoClawClient:
    """
    Returns the appropriate client based on environment:
      - API key present → NeMoGuardrailsClient (Colang agent + NIM fallback)
      - API key absent  → MockNeMoClawClient (offline, deterministic)
    """
    if not settings.NVIDIA_API_KEY:
        logger.warning(
            "NVIDIA_API_KEY not set — using MockNeMoClawClient. "
            "Set NVIDIA_API_KEY in .env to enable live NIM inference."
        )
        return MockNeMoClawClient()
    return NeMoGuardrailsClient()

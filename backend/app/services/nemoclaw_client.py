"""
NeMoClaw Client — the orchestration adapter between FastAPI and the NVIDIA NIM endpoint.

Architecture role:
  FastAPI routes → AgentRunner → NeMoClawClient → NVIDIA NIM inference API
                                       ↑
                              enforces Pydantic schema,
                              handles retries, parses output

What NeMoClaw is responsible for here:
  • Prompt compilation (system + user template fusion)
  • Inference request execution against the NIM endpoint
  • Structured-output enforcement: the LLM is instructed to return strict JSON;
    this client validates and coerces the raw string into a HypothesisOutput object
  • Retry logic on malformed responses (up to MAX_RETRIES attempts)
  • Token-level metadata extraction (usage, latency)

NIM compatibility:
  NVIDIA NIM exposes an OpenAI-compatible /chat/completions endpoint.
  We use the `openai` SDK pointed at NIM_BASE_URL so the same client code
  works locally (mock key) and on Brev (real NVIDIA_API_KEY + NIM endpoint).

If you later get access to the native NeMoClaw Python SDK, replace the
`_call_nim()` method body — the rest of the class stays identical.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

import httpx
from pydantic import ValidationError

from app.core.config import get_settings
from app.schemas.pydantic import HypothesisOutput

logger = logging.getLogger(__name__)

settings = get_settings()

MAX_RETRIES = 2
RETRY_DELAY_S = 1.5


class NeMoClawError(Exception):
    """Raised when the NeMoClaw layer cannot produce a valid structured output."""


class NeMoClawClient:
    """
    Thin adapter that makes NIM look like a typed inference service.

    Usage:
        client = NeMoClawClient()
        output: HypothesisOutput = client.run(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_msg,
            response_model=HypothesisOutput,
        )
    """

    def __init__(self) -> None:
        self._base_url = settings.NIM_BASE_URL.rstrip("/")
        self._api_key = settings.NVIDIA_API_KEY
        self._model = settings.NIM_MODEL
        self._max_tokens = settings.MAX_SYNTHESIS_TOKENS
        self._temperature = settings.LLM_TEMPERATURE

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[HypothesisOutput],
    ) -> tuple[HypothesisOutput, str]:
        """
        Execute the inference request and return (structured_output, raw_json_string).

        Retries up to MAX_RETRIES times if the LLM returns malformed JSON.
        Raises NeMoClawError if all retries are exhausted.
        """
        last_error: Optional[Exception] = None
        raw_text = ""

        for attempt in range(1, MAX_RETRIES + 2):  # +2 = initial try + MAX_RETRIES
            try:
                raw_text = self._call_nim(system_prompt, user_prompt)
                logger.debug("NeMoClaw raw response (attempt %d): %s", attempt, raw_text[:300])
                structured = self._parse_and_validate(raw_text, response_model)
                logger.info(
                    "NeMoClaw inference succeeded on attempt %d (confidence=%s)",
                    attempt,
                    structured.confidence_score,
                )
                return structured, raw_text

            except (NeMoClawError, ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "NeMoClaw attempt %d failed: %s — %s",
                    attempt,
                    type(exc).__name__,
                    str(exc)[:120],
                )
                if attempt <= MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S)

        raise NeMoClawError(
            f"NeMoClaw failed to produce valid structured output after "
            f"{MAX_RETRIES + 1} attempts. Last error: {last_error}. "
            f"Raw text snippet: {raw_text[:200]}"
        )

    # ── Private methods ───────────────────────────────────────────────────────

    def _call_nim(self, system_prompt: str, user_prompt: str) -> str:
        """
        POST to the NIM /chat/completions endpoint.
        Returns the raw assistant message string.
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            # NIM supports response_format for some models; uncomment if your
            # deployed model supports it for guaranteed JSON mode:
            # "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise NeMoClawError(
                f"NIM HTTP error {exc.response.status_code}: {exc.response.text[:300]}"
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
        """
        Extract JSON from the raw string (handles markdown fences or leading text)
        and validate it against the Pydantic response_model.
        """
        # Strip markdown code fences the model might wrap around JSON
        fence_pattern = r"```(?:json)?\s*([\s\S]+?)\s*```"
        match = re.search(fence_pattern, raw)
        json_str = match.group(1) if match else raw.strip()

        # Further strip any leading prose before the opening brace
        brace_idx = json_str.find("{")
        if brace_idx > 0:
            json_str = json_str[brace_idx:]

        parsed = json.loads(json_str)
        return model.model_validate(parsed)


# ── Mock fallback (used when NVIDIA_API_KEY is absent) ────────────────────────

class MockNeMoClawClient(NeMoClawClient):
    """
    Deterministic mock that returns a realistic-looking hypothesis without
    hitting any external API.  Activated automatically when NVIDIA_API_KEY
    is missing — keeps the demo runnable locally/offline while preserving
    the full orchestration pipeline.
    """

    _MOCK_RESPONSES: dict[str, dict] = {
        "efficient llm routing": {
            "gap_identified": (
                "Existing routing systems make per-token or static routing decisions "
                "that ignore inter-token semantic continuity and cannot adapt to "
                "distribution shift at inference time, leading to expert collapse and "
                "degraded performance on out-of-distribution queries."
            ),
            "proposed_architecture": (
                "SeqRouter: a sequence-level routing transformer that encodes a sliding "
                "window of hidden states via a lightweight 2-layer cross-attention module "
                "(4 heads, d=128). The router outputs a soft gate vector over a pool of "
                "N=4 draft models ranked by capacity, updated online via an EMA of "
                "accept-rate feedback from speculative verification. Trained end-to-end "
                "with a contrastive routing loss against human preference data."
            ),
            "evaluation_metric": "MMLU accuracy degradation vs. routing latency (ms) on MT-Bench",
            "confidence_score": "HIGH",
        },
        "vision transformer robustness": {
            "gap_identified": (
                "ViTs concentrate attention on a small subset of salient tokens and have "
                "no dynamic mechanism to redistribute attention when those tokens are "
                "corrupted, making them exploitable at the token level despite headline "
                "robustness figures."
            ),
            "proposed_architecture": (
                "AdaptiveShield-ViT: adds an uncertainty-aware attention re-routing layer "
                "after each Transformer block. When per-token prediction entropy exceeds a "
                "learned threshold, the layer re-weights attention to the top-K globally "
                "consistent tokens via a sparse softmax. Trained with an adversarial "
                "curriculum on Patch-Fool perturbations alongside standard supervision."
            ),
            "evaluation_metric": "mCE on ImageNet-C and ASR on Patch-Fool adversarial set",
            "confidence_score": "MEDIUM",
        },
        "llm hallucination detection": {
            "gap_identified": (
                "Multi-sample consistency methods fail on systematic hallucinations where "
                "all sampled responses agree on a false claim. Single-pass factual "
                "verification methods require curated knowledge bases and do not model "
                "claim interdependency or source trustworthiness."
            ),
            "proposed_architecture": (
                "ClaimGraph: a two-stage pipeline that (1) decomposes generated text into "
                "atomic claims using a fine-tuned T5-Claim extractor and constructs a "
                "directed dependency graph between claims, then (2) scores each claim "
                "with a retrieval-augmented verifier that weights retrieved passages by "
                "source credibility via a trained passage-trust classifier. Uncertainty "
                "propagates through the graph via loopy belief propagation."
            ),
            "evaluation_metric": "F1 hallucination detection rate on FActScore benchmark",
            "confidence_score": "HIGH",
        },
        "protein structure prediction": {
            "gap_identified": (
                "Current models produce single static structures and yield poorly "
                "calibrated confidence scores for disordered regions and novel covalent "
                "binders, limiting utility for conformational ensemble analysis and "
                "covalent drug discovery."
            ),
            "proposed_architecture": (
                "EnsembleFlow: a flow-matching generative model conditioned on ESMFold "
                "embeddings that samples a distribution over backbone conformations rather "
                "than a single structure. A covalent-bond topology head is jointly trained "
                "on CovalentPDB to predict reactive atom pairs, enabling de novo covalent "
                "inhibitor pose generation without requiring pre-specified bond topology."
            ),
            "evaluation_metric": "TM-score diversity on CATH conformational ensemble benchmark",
            "confidence_score": "MEDIUM",
        },
    }

    _DEFAULT_KEY = "efficient llm routing"

    def _call_nim(self, system_prompt: str, user_prompt: str) -> str:  # type: ignore[override]
        logger.info("MockNeMoClawClient: returning deterministic hypothesis (no API key set).")
        # Infer which mock to use from the user prompt
        for key in self._MOCK_RESPONSES:
            if key in user_prompt.lower():
                return json.dumps(self._MOCK_RESPONSES[key])
        return json.dumps(self._MOCK_RESPONSES[self._DEFAULT_KEY])


def get_nemoclaw_client() -> NeMoClawClient:
    """
    Factory that returns the real or mock client based on environment.
    Import this function everywhere rather than constructing the client directly.
    """
    if not settings.NVIDIA_API_KEY:
        logger.warning(
            "NVIDIA_API_KEY not set — using MockNeMoClawClient. "
            "Set NVIDIA_API_KEY in .env to enable live NIM inference."
        )
        return MockNeMoClawClient()
    return NeMoClawClient()

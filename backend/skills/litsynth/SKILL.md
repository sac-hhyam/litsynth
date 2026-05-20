---
name: litsynth
version: "1.0.0"
description: Fetches academic papers from OpenAlex and synthesises a structured research hypothesis using NVIDIA NIM.
entrypoint: python synthesise.py
dependencies:
  - httpx
env:
  - NVIDIA_API_KEY
  - NIM_BASE_URL
  - NIM_MODEL
---

# LitSynth Synthesis Skill

Fetches recent academic papers from OpenAlex and synthesises a structured
research hypothesis using NVIDIA NIM (Nemotron).

## Usage

```
python synthesise.py <topic>
```

Returns a JSON object on stdout:

```json
{
  "gap_identified": "...",
  "proposed_architecture": "...",
  "evaluation_metric": "...",
  "confidence_score": "HIGH|MEDIUM|LOW",
  "topic": "...",
  "papers_used": 4,
  "source": "openalex"
}
```

## Dependencies

- httpx

## Environment Variables

- `NVIDIA_API_KEY` — NVIDIA NIM API key
- `NIM_BASE_URL` — defaults to `https://integrate.api.nvidia.com/v1`
- `NIM_MODEL` — defaults to `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`

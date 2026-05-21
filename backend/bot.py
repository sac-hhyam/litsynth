"""
LitSynth Discord Bot — NeMoClaw Interface Layer
================================================

Provides a !synthesize <topic> command that executes the synthesis skill
inside the lit-synth-sandbox OpenShell sandbox via NeMoClaw's exec bridge,
then posts the results back to Discord.

Architecture:
  Discord !synthesize
      │
      ▼
  bot.py (host — this file)
      │  nemoclaw lit-synth-sandbox exec --no-tty -- python synthesise.py <topic>
      ▼
  lit-synth-sandbox (OpenShell sandbox — policy-enforced)
      │  OpenAlex fetch → NIM /chat/completions → JSON stdout
      ▼
  bot.py parses stdout JSON → formats Markdown → posts to Discord

Environment variables (add to .env or export before running):
    DISCORD_BOT_TOKEN       — from Discord Developer Portal (Bot > Token)
    NEMOCLAW_SANDBOX        — sandbox name (default: lit-synth-sandbox)
    SYNTHESIS_TIMEOUT       — seconds to wait for sandbox exec (default: 120)

Run:
    cd backend
    source venv/bin/activate
    pip install discord.py
    python bot.py
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

import discord
from discord.ext import commands

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("litsynth.bot")

# ── Config ────────────────────────────────────────────────────────────────────
DISCORD_BOT_TOKEN  = os.environ.get("DISCORD_BOT_TOKEN", "")
NEMOCLAW_SANDBOX   = os.environ.get("NEMOCLAW_SANDBOX", "lit-synth-sandbox")
SYNTHESIS_TIMEOUT  = int(os.environ.get("SYNTHESIS_TIMEOUT", "120"))
# Path to the skill script inside the sandbox workspace
SKILL_SCRIPT       = "synthesise.py"
SKILL_WORKDIR      = "/sandbox/.openclaw/skills/litsynth"
# Discord message length hard limit
DISCORD_MAX_CHARS  = 2000

if not DISCORD_BOT_TOKEN:
    logger.error(
        "DISCORD_BOT_TOKEN is not set. "
        "Export it or add it to .env before running bot.py."
    )
    sys.exit(1)

# ── Bot setup ─────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True   # required for reading command arguments

bot = commands.Bot(command_prefix="!", intents=intents)


# ── Sandbox exec helper ───────────────────────────────────────────────────────

def fetch_papers_on_host(topic: str) -> list[dict]:
    """
    Fetch papers from OpenAlex on the host (unrestricted network).
    The sandbox proxy blocks api.openalex.org, so we fetch here and
    pass the results into the sandbox via --context.
    """
    import httpx, textwrap

    def _reconstruct_abstract(inv_idx: dict | None) -> str:
        if not inv_idx:
            return ""
        max_pos = max(pos for positions in inv_idx.values() for pos in positions)
        tokens: list[str] = [""] * (max_pos + 1)
        for word, positions in inv_idx.items():
            for pos in positions:
                tokens[pos] = word
        return " ".join(t for t in tokens if t)

    params = {
        "search":   topic,
        "per_page": 4,
        "select":   "display_name,authorships,abstract_inverted_index,publication_year",
        "mailto":   "litsynth@demo.nvaitc.ai",
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.get("https://api.openalex.org/works", params=params)
    resp.raise_for_status()

    papers = []
    for w in resp.json().get("results", []):
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
        if not abstract.strip():
            continue
        names = [
            a["author"]["display_name"]
            for a in w.get("authorships", [])
            if a.get("author", {}).get("display_name")
        ]
        author_str = (", ".join(names[:3]) + " et al." if len(names) > 3
                      else ", ".join(names))
        year = w.get("publication_year")
        if year:
            author_str = f"{author_str}, {year}"
        papers.append({
            "title":    w.get("display_name", "Untitled"),
            "authors":  author_str,
            "abstract": textwrap.shorten(abstract, width=800, placeholder="..."),
        })

    if not papers:
        raise RuntimeError(f"OpenAlex returned no papers with abstracts for: '{topic}'")

    logger.info("Host fetched %d papers for '%s' from OpenAlex", len(papers), topic)
    return papers


def run_synthesis_in_sandbox(topic: str) -> dict:
    """
    Execute the synthesis skill inside the OpenShell sandbox via NeMoClaw exec.

    Uses:
        nemoclaw <sandbox> exec --no-tty --workdir /workspace
            -- python synthesise.py <topic>

    Returns the parsed JSON dict emitted by synthesise.py on stdout.
    Raises RuntimeError on subprocess failure or JSON parse error.
    """
    # Fetch papers on the host — sandbox proxy blocks api.openalex.org
    papers = fetch_papers_on_host(topic)
    papers_json = json.dumps(papers)

    cmd = [
        "nemoclaw", NEMOCLAW_SANDBOX, "exec",
        "--no-tty",
        "--workdir", SKILL_WORKDIR,
        "--",
        "python3", SKILL_SCRIPT,
        "--context", papers_json,
    ] + topic.split()

    logger.info("Dispatching to sandbox: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SYNTHESIS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Sandbox exec timed out after {SYNTHESIS_TIMEOUT}s. "
            "Try a shorter topic or increase SYNTHESIS_TIMEOUT."
        )
    except FileNotFoundError:
        raise RuntimeError(
            "nemoclaw binary not found. "
            "Ensure NeMoClaw is installed and on PATH before running bot.py."
        )

    if proc.returncode != 0:
        stderr = proc.stderr.strip()[:500]
        raise RuntimeError(
            f"Sandbox exec exited with code {proc.returncode}. "
            f"stderr: {stderr}"
        )

    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError("Sandbox exec produced no output on stdout.")

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Sandbox output was not valid JSON ({exc}). "
            f"Raw output: {stdout[:300]}"
        )

    if "error" in result:
        raise RuntimeError(f"Synthesis skill reported error: {result['error']}")

    return result


# ── Formatting ────────────────────────────────────────────────────────────────

def format_hypothesis_markdown(result: dict) -> str:
    """Render the synthesis result as a Discord-ready Markdown string."""
    confidence_emoji = {
        "HIGH":   "🟢",
        "MEDIUM": "🟡",
        "LOW":    "🔴",
    }.get(result.get("confidence_score", "").upper(), "⚪")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"## 🔬 Research Hypothesis — *{result.get('topic', 'Unknown Topic')}*\n"
        f"> Generated {ts} via `lit-synth-sandbox` · "
        f"{result.get('papers_used', '?')} papers · "
        f"source: `{result.get('source', 'unknown')}`\n\n"
        f"### 🕳️ Research Gap\n"
        f"{result.get('gap_identified', '—')}\n\n"
        f"### 🏗️ Proposed Architecture\n"
        f"{result.get('proposed_architecture', '—')}\n\n"
        f"### 📏 Evaluation Metric\n"
        f"`{result.get('evaluation_metric', '—')}`\n\n"
        f"### {confidence_emoji} Confidence Score\n"
        f"**{result.get('confidence_score', '—')}**\n"
    )


async def send_or_upload(
    ctx: commands.Context,
    content: str,
    topic: str,
) -> None:
    """
    Post content to Discord.

    If content fits within DISCORD_MAX_CHARS, send it directly.
    Otherwise upload it as a .md file attachment so nothing is truncated.
    """
    if len(content) <= DISCORD_MAX_CHARS:
        await ctx.send(content)
        return

    # Content too long — upload as markdown file
    filename = f"hypothesis_{topic[:40].replace(' ', '_')}.md"
    file_obj  = io.BytesIO(content.encode("utf-8"))
    discord_file = discord.File(fp=file_obj, filename=filename)
    await ctx.send(
        content=(
            f"📎 Hypothesis is too long for a single message — "
            f"uploaded as **{filename}**"
        ),
        file=discord_file,
    )


# ── Commands ──────────────────────────────────────────────────────────────────

@bot.command(name="synthesize", aliases=["synth", "s"])
async def synthesize(ctx: commands.Context, *, topic: str = "") -> None:
    """
    !synthesize <topic>

    Runs a full literature synthesis inside the NeMoClaw OpenShell sandbox
    and posts the structured hypothesis back to this channel.

    Examples:
        !synthesize efficient LLM routing
        !synthesize vision transformer robustness
        !synthesize protein structure prediction
        !synth hallucination detection in large language models
    """
    if not topic:
        await ctx.send(
            "❌ Please provide a research topic.\n"
            "Usage: `!synthesize <topic>`\n"
            "Example: `!synthesize efficient LLM routing`"
        )
        return

    # Acknowledge immediately so the user knows the bot received the command
    thinking_msg = await ctx.send(
        f"⚙️ Running synthesis for **{topic}** inside `{NEMOCLAW_SANDBOX}`…\n"
        f"_(fetching papers → NIM inference → parsing — ~30–60s)_"
    )

    try:
        # Run the blocking subprocess in a thread so the event loop stays alive
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            run_synthesis_in_sandbox,
            topic,
        )
    except RuntimeError as exc:
        await thinking_msg.edit(
            content=f"❌ Synthesis failed: {exc}"
        )
        logger.error("Synthesis failed for topic '%s': %s", topic, exc)
        return

    output = format_hypothesis_markdown(result)

    # Delete the "thinking…" message and post the real result
    await thinking_msg.delete()
    await send_or_upload(ctx, output, topic)
    logger.info("Synthesis complete for topic '%s'", topic)


@bot.command(name="sandbox")
async def sandbox_status(ctx: commands.Context) -> None:
    """!sandbox — Show the current NeMoClaw sandbox status."""
    try:
        proc = subprocess.run(
            ["nemoclaw", NEMOCLAW_SANDBOX, "status"],
            capture_output=True, text=True, timeout=15,
        )
        status_text = (proc.stdout or proc.stderr).strip()[:1800]
    except Exception as exc:
        status_text = f"Error fetching sandbox status: {exc}"

    await ctx.send(f"```\n{status_text}\n```")


@bot.command(name="policy")
async def show_policy(ctx: commands.Context) -> None:
    """!policy — List active security policies on the sandbox."""
    try:
        proc = subprocess.run(
            ["nemoclaw", NEMOCLAW_SANDBOX, "policy-list"],
            capture_output=True, text=True, timeout=15,
        )
        policy_text = (proc.stdout or proc.stderr).strip()[:1800]
    except Exception as exc:
        policy_text = f"Error fetching policies: {exc}"

    await ctx.send(f"```\n{policy_text}\n```")


# ── Bot lifecycle events ───────────────────────────────────────────────────────

@bot.event
async def on_ready() -> None:
    logger.info(
        "LitSynth bot online as %s (id=%s) | sandbox=%s",
        bot.user, bot.user.id, NEMOCLAW_SANDBOX,
    )
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="research gaps | !synthesize <topic>",
        )
    )


@bot.event
async def on_command_error(ctx: commands.Context, error: Exception) -> None:
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ Missing argument: `{error.param.name}`")
    elif isinstance(error, commands.CommandNotFound):
        pass  # silently ignore unknown commands
    else:
        logger.error("Unhandled command error: %s", error)
        await ctx.send(f"❌ Unexpected error: {error}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting LitSynth Discord bot (sandbox: %s)", NEMOCLAW_SANDBOX)
    bot.run(DISCORD_BOT_TOKEN, log_handler=None)

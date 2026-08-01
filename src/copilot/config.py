"""Single place where environment configuration is read and validated.

Every script imports `settings` from here rather than touching os.environ, so a
missing credential fails once, loudly, with instructions -- instead of surfacing
as an opaque 401 from somewhere deep in the SDK.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
KB_DIR = DATA_DIR / "kb"

load_dotenv(REPO_ROOT / ".env")


class ConfigError(RuntimeError):
    """Raised when required configuration is absent."""


@dataclass(frozen=True)
class Settings:
    arize_api_key: str
    arize_space_id: str
    arize_space_name: str
    arize_project_name: str
    arize_region: str | None
    arize_otlp_endpoint: str
    deepseek_api_key: str
    deepseek_base_url: str
    arize_ai_integration_id: str | None
    prompt_version: str
    impl: str = "sdk"

    @property
    def app_url(self) -> str:
        return f"https://app.arize.com/organizations/-/spaces/{self.arize_space_id}"


_REQUIRED = {
    "ARIZE_API_KEY": "Arize Space Settings -> API Keys",
    "ARIZE_SPACE_ID": "Arize Space Settings -> Space ID",
    "ARIZE_SPACE_NAME": "the space's display name, shown in the Arize sidebar",
    "DEEPSEEK_API_KEY": "https://platform.deepseek.com -> API keys",
}

# DeepSeek speaks the OpenAI chat-completions protocol, so the whole app talks
# to it through the `openai` SDK with a different base URL. That is also why
# OpenAIInstrumentor produces correct LLM spans for it.
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# `thinking` is a DeepSeek extension rather than part of the OpenAI schema, so
# it travels in extra_body -- the openai SDK raises TypeError on unrecognised
# keyword arguments. It is enabled by default on V4, and turning it off is
# sometimes required rather than merely cheaper: a forced `tool_choice` is
# rejected while thinking is on, which is what the Phoenix judge in poc/04
# needs, and a small max_tokens budget gets consumed entirely by the CoT,
# which is what the intent router in agent.py needs.
THINKING_ON: dict = {"thinking": {"type": "enabled"}}
THINKING_OFF: dict = {"thinking": {"type": "disabled"}}


# --------------------------------------------------------------------------
# Region
#
# Arize keys are region-scoped, and a key from one region hitting another
# returns an authorization failure, not a redirect -- on the trace exporter it
# surfaces as "unable to validate authorization from span".
#
# Region has to reach *two* independent places, which is the trap: the OTel
# exporter takes a collector URL (env ARIZE_COLLECTOR_ENDPOINT), while the
# platform client takes a region enum (env ARIZE_REGION). Setting only the
# first fixes tracing and leaves every export/dataset/experiment call pointed
# at the default region. So ARIZE_REGION is the single knob here, and the
# endpoint is derived from it.
# --------------------------------------------------------------------------

DEFAULT_OTLP_ENDPOINT = "https://otlp.arize.com/v1"

_REGION_OTLP_ENDPOINTS = {
    "eu-west-1a": "https://otlp.eu-west-1a.arize.com/v1",
    "ca-central-1a": "https://otlp.ca-central-1a.arize.com/v1",
}


def _resolve_otlp_endpoint(region: str | None) -> str:
    """Collector URL for a region, with an explicit override winning.

    Both env names are honoured: ARIZE_COLLECTOR_ENDPOINT is what the Arize SDK
    itself reads, and ARIZE_OTLP_ENDPOINT is the name people reach for first.
    """
    explicit = os.getenv("ARIZE_COLLECTOR_ENDPOINT") or os.getenv("ARIZE_OTLP_ENDPOINT")
    if explicit:
        return explicit
    return _REGION_OTLP_ENDPOINTS.get(region or "", DEFAULT_OTLP_ENDPOINT)


# Two engines produce the traffic: the hand-written tool loop over the raw
# `openai` SDK ("sdk", the default) and the LangGraph/LangChain build of the
# same copilot ("langgraph"). Same prompts, same tools, same corpus, same
# seeded failures -- COPILOT_IMPL picks which one `copilot.agent` dispatches to.
_IMPLS = ("sdk", "langgraph")


def _resolve_impl() -> str:
    impl = os.getenv("COPILOT_IMPL", "sdk").strip().lower()
    if impl not in _IMPLS:
        raise ConfigError(
            f"COPILOT_IMPL={impl!r} is not one of {', '.join(_IMPLS)}."
        )
    return impl


def load_settings() -> Settings:
    missing = [f"  {name:<22} ({where})" for name, where in _REQUIRED.items() if not os.getenv(name)]
    if missing:
        raise ConfigError(
            "Missing required environment variables:\n"
            + "\n".join(missing)
            + "\n\nCopy .env.example to .env and fill these in."
        )
    region = os.getenv("ARIZE_REGION") or None
    impl = _resolve_impl()
    project_name = os.getenv("ARIZE_PROJECT_NAME", "nimbus-support-copilot")
    # The LangGraph engine writes to its own project so the two engines'
    # traffic never mixes -- traces, online eval tasks, monitors, and the
    # per-project review queue all follow the project name.
    if impl == "langgraph" and not project_name.endswith("-lg"):
        project_name = f"{project_name}-lg"
    return Settings(
        arize_api_key=os.environ["ARIZE_API_KEY"],
        arize_space_id=os.environ["ARIZE_SPACE_ID"],
        arize_space_name=os.environ["ARIZE_SPACE_NAME"],
        arize_project_name=project_name,
        arize_region=region,
        arize_otlp_endpoint=_resolve_otlp_endpoint(region),
        deepseek_api_key=os.environ["DEEPSEEK_API_KEY"],
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
        arize_ai_integration_id=os.getenv("ARIZE_AI_INTEGRATION_ID") or None,
        prompt_version=os.getenv("COPILOT_PROMPT_VERSION", "v1"),
        impl=impl,
    )


def platform_client(settings: Settings):
    """The Arize platform client, region-matched to the space.

    One definition, because getting it wrong is quiet: keys are region-scoped
    and a cross-region call fails authorization rather than redirecting. A
    caller that builds `ArizeClient(api_key=...)` directly works fine in the
    default region and fails everywhere else.
    """
    from arize.client import ArizeClient

    if settings.arize_region:
        from arize.regions import Region

        return ArizeClient(api_key=settings.arize_api_key, region=Region(settings.arize_region))
    return ArizeClient(api_key=settings.arize_api_key)


def settings_or_exit() -> Settings:
    """Entry point for CLI scripts: print the fix and exit 1 rather than traceback."""
    try:
        return load_settings()
    except ConfigError as exc:
        print(f"\n[config] {exc}\n", file=sys.stderr)
        raise SystemExit(1) from None


# The model under observation. The DeepSeek V4 line is two models -- pro
# (1.6T/49B) and flash (284B/13B) -- both with 1M context and both able to run
# thinking or non-thinking. The older `deepseek-chat` / `deepseek-reasoner`
# aliases were retired on 2026-07-24 and now 404; don't reintroduce them.
AGENT_MODEL = os.getenv("COPILOT_AGENT_MODEL", "deepseek-v4-pro")

# The intent router is a one-word classification, so it gets the cheap model
# with thinking off (see agent.py) -- flash is ~6x smaller on active params.
ROUTER_MODEL = os.getenv("COPILOT_ROUTER_MODEL", "deepseek-v4-flash")

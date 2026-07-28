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


def load_settings() -> Settings:
    missing = [f"  {name:<22} ({where})" for name, where in _REQUIRED.items() if not os.getenv(name)]
    if missing:
        raise ConfigError(
            "Missing required environment variables:\n"
            + "\n".join(missing)
            + "\n\nCopy .env.example to .env and fill these in."
        )
    region = os.getenv("ARIZE_REGION") or None
    return Settings(
        arize_api_key=os.environ["ARIZE_API_KEY"],
        arize_space_id=os.environ["ARIZE_SPACE_ID"],
        arize_space_name=os.environ["ARIZE_SPACE_NAME"],
        arize_project_name=os.getenv("ARIZE_PROJECT_NAME", "nimbus-support-copilot"),
        arize_region=region,
        arize_otlp_endpoint=_resolve_otlp_endpoint(region),
        deepseek_api_key=os.environ["DEEPSEEK_API_KEY"],
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
        arize_ai_integration_id=os.getenv("ARIZE_AI_INTEGRATION_ID") or None,
        prompt_version=os.getenv("COPILOT_PROMPT_VERSION", "v1"),
    )


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

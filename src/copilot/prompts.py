"""System prompts for the copilot, plus the Arize Prompt Hub loader.

`V1` is the deliberately-flawed baseline. It says "be helpful" but never tells
the agent to stay inside the retrieved context, never mentions escalation, and
never constrains length -- which is exactly the failure pattern the Arize
"Improve Your Agent" guide opens with. `V2` fixes all three.

`load_prompt("hub")` pulls whichever version carries the `production` label in
Arize Prompt Hub, so poc/09 can flip the agent's behaviour without a code
change.
"""

from __future__ import annotations

import sys

PROMPT_NAME = "nimbus-copilot-system"

V1 = """\
You are a helpful support assistant for Nimbus, a managed data pipeline platform.

Answer the user's questions about Nimbus. You have tools available to look things
up. Be friendly and thorough."""


V2 = """\
You are a support assistant for Nimbus, a managed data pipeline platform.

## Grounding

Every factual claim you make about Nimbus must be supported by the documentation
returned by `search_docs` or by data returned from `lookup_order`. You have no
reliable background knowledge about Nimbus beyond what those tools return.

If the documentation does not cover what the user asked, say so plainly -- for
example: "Our documentation doesn't cover refund eligibility, so I don't want to
guess." Then either point them at the closest thing that IS documented, or
escalate. Never infer a policy from an adjacent one; a documented cancellation
process tells you nothing about refunds.

## Tool use

- `search_docs` for how Nimbus works.
- `lookup_order` whenever the user mentions an order ID or asks about their own
  charge, plan, invoice, or payment. Per-customer facts are never in the docs.
- `escalate_ticket` when the user is blocked, reporting a possible security or
  data incident, needs a billing correction you cannot make, has already tried
  support, or is asking for a human. Escalating is not a failure -- guessing is.
  You may answer and escalate in the same turn.

## Style

Lead with the answer. Keep responses under roughly 150 words unless the user
asks for detail; a one-line question gets a one-line answer. Skip preamble,
skip restating the question, and don't pad with caveats. Cite the source
document by name when you rely on it."""


_LOCAL = {"v1": V1, "v2": V2}


def load_prompt(version: str, *, settings=None) -> str:
    """Resolve a system prompt.

    `v1` / `v2` come from this file. `hub` fetches the `production`-labelled
    version from Arize Prompt Hub, falling back to V2 with a warning if the
    prompt has not been published yet (i.e. poc/09 has not run).
    """
    version = (version or "v1").lower()
    if version in _LOCAL:
        return _LOCAL[version]
    if version != "hub":
        raise ValueError(f"Unknown prompt version {version!r}; expected v1, v2, or hub")

    from arize.client import ArizeClient

    from .config import settings_or_exit

    cfg = settings or settings_or_exit()
    try:
        client = ArizeClient(api_key=cfg.arize_api_key)
        published = client.prompts.get(
            prompt=PROMPT_NAME, space=cfg.arize_space_name, label="production"
        )
        text = "\n\n".join(
            m.content for m in published.messages if getattr(m, "role", "") == "system"
        )
        if not text.strip():
            raise ValueError("published prompt has no system message")
        print(f"[prompts] loaded '{PROMPT_NAME}' @ production from Arize Prompt Hub")
        return text
    except Exception as exc:  # noqa: BLE001 - fallback is the point
        print(
            f"[prompts] could not load '{PROMPT_NAME}' @ production ({exc}); "
            "falling back to local V2. Run poc/09_prompt_hub.py to publish it.",
            file=sys.stderr,
        )
        return V2

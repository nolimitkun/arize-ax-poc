#!/usr/bin/env python
"""Step 09 -- Improve: publish the winning prompt, then load it at runtime.

The point of Prompt Hub is that the prompt stops being a string in your
repository and becomes a versioned, labelled artifact. `production` is a moving
pointer: promote a new version and running services pick it up on their next
fetch, and roll back by moving the label back.

`copilot.prompts.load_prompt("hub")` is the consumer side of that.

Docs: https://arize.com/docs/ax/improve/build-a-prompt
      https://arize.com/docs/ax/concepts/prompts/prompt-hub
      https://arize.com/docs/ax/concepts/prompts/versioning-and-tags
      https://arize.com/docs/ax/concepts/prompts/loading-in-applications
"""

from __future__ import annotations

import typer

from _common import arize_client, console, done, header, look_at, table

app = typer.Typer(add_completion=False)

PRODUCTION = "production"


def version_id(obj) -> str:
    for attr in ("version_id", "id"):
        value = getattr(obj, attr, None)
        if value:
            return str(value)
    version = getattr(obj, "version", None)
    return str(getattr(version, "id", "")) if version else ""


@app.command()
def main(
    promote: str = typer.Option("v2", help="Which local version to label `production`"),
    verify: bool = typer.Option(True, help="Run one live turn against the published prompt"),
) -> None:
    settings = header(
        "09",
        "Improve: publish to Prompt Hub, label, and load at runtime",
        "build-a-prompt · prompt-hub · versioning-and-tags · loading-in-applications",
    )

    from arize.prompts.types import InputVariableFormat, LLMMessage, LlmProvider, MessageRole

    from copilot.prompts import PROMPT_NAME, V1, V2

    # Arize has no first-class DeepSeek provider. CUSTOM is the honest label for
    # "speaks the OpenAI protocol, but isn't OpenAI" -- the endpoint itself is
    # bound by the space's AI integration (step 05), not by the prompt record.
    PROVIDER = LlmProvider.CUSTOM
    MODEL = "deepseek-v4-pro"

    client = arize_client(settings)
    space = settings.arize_space_name

    def message(text: str) -> LLMMessage:
        return LLMMessage(role=MessageRole.SYSTEM, content=text)

    # ---- v1: create the prompt -------------------------------------------
    console.print("[bold]Publishing v1[/bold] (the flawed baseline)")
    try:
        v1 = client.prompts.create(
            space=space,
            name=PROMPT_NAME,
            commit_message="v1 baseline: helpful, but ungrounded and never escalates",
            description="System prompt for the Nimbus support copilot.",
            input_variable_format=InputVariableFormat.NONE,
            provider=PROVIDER,
            model=MODEL,
            messages=[message(V1)],
        )
        v1_id = version_id(v1)
        console.print(f"  created [bold]{PROMPT_NAME}[/bold] v1 (version {v1_id})")
    except Exception as exc:  # noqa: BLE001 - re-runs: prompt already exists
        console.print(f"  [yellow]exists already ({exc}); continuing[/yellow]")
        v1_id = version_id(client.prompts.get(prompt=PROMPT_NAME, space=space))

    # ---- v2: new immutable version ---------------------------------------
    console.print("\n[bold]Publishing v2[/bold] (grounded, escalates, concise)")
    v2 = client.prompts.create_version(
        prompt=PROMPT_NAME,
        space=space,
        commit_message=(
            "v2: require grounding in retrieved docs, admit gaps, escalate when "
            "blocked, cap length. Beats v1 on the copilot-failures dataset."
        ),
        input_variable_format=InputVariableFormat.NONE,
        provider=PROVIDER,
        model=MODEL,
        messages=[message(V2)],
    )
    v2_id = version_id(v2)
    console.print(f"  created version [bold]{v2_id}[/bold]")

    # ---- label -----------------------------------------------------------
    target = v2_id if promote == "v2" else v1_id
    client.prompts.set_labels(version_id=target, labels=[PRODUCTION])
    console.print(f"\n[green]Labelled version {target} as `{PRODUCTION}`[/green]")

    versions = client.prompts.list_versions(prompt=PROMPT_NAME, space=space)
    items = getattr(versions, "data", None) or getattr(versions, "prompt_versions", [])
    table(
        f"{PROMPT_NAME} versions",
        ["version id", "commit message", "labels"],
        [
            [
                str(getattr(v, "id", "?"))[:24],
                str(getattr(v, "commit_message", ""))[:52],
                ", ".join(getattr(v, "labels", []) or []) or "-",
            ]
            for v in items
        ],
    )

    # ---- consume it ------------------------------------------------------
    if verify:
        console.print("\n[bold]Loading the published prompt at runtime[/bold]")
        from copilot.agent import run_turn
        from copilot.prompts import load_prompt
        from copilot.tracing import flush, init_tracing

        published = load_prompt("hub", settings=settings)
        console.print(f"  fetched {len(published)} chars from Prompt Hub")

        init_tracing(settings)
        probe = "If I cancel mid-month, do I get a prorated refund for the unused days?"
        console.print(f"\n  probe: [dim]{probe}[/dim]")
        result = run_turn(
            probe,
            settings=settings,
            prompt_version="hub",
            tags=["prompt-hub-verify"],
        )
        flush()
        console.print(f"  answer: {result.answer[:220]}…\n")

        hedged = any(
            m in result.answer.lower()
            for m in ("doesn't cover", "does not cover", "not documented", "don't want to guess")
        )
        if hedged:
            console.print(
                "[bold green]The agent declined to invent a refund policy.[/bold green] "
                "That behaviour came from Prompt Hub, not from this repository — "
                "no code changed between v1 and now.\n"
            )
        else:
            console.print(
                "[yellow]The agent still answered confidently.[/yellow] Check that "
                f"`{PRODUCTION}` points at the v2 version above.\n"
            )

    look_at(
        f"Prompt Hub → {PROMPT_NAME}. Two versions, with commit messages and a diff.",
        f"The `{PRODUCTION}` label on v2 — move it back to v1 and re-run this script "
        "to watch the agent regress. That is your rollback path.",
        "Open the prompt in Playground and run it against the copilot-failures dataset.",
    )
    done("poc/10_monitors.py — monitors and a dashboard over the eval metrics")


if __name__ == "__main__":
    app()

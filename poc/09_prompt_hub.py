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


def list_versions(client, prompt_name: str, space: str) -> list:
    """Every published version, newest first, or [] if the prompt is new.

    The prompt name is a parameter rather than read from the module: it is
    imported inside main(), so a global reference here raises NameError, and the
    except below would report that as "no versions yet" and quietly republish
    everything. The reason is printed for the same reason.
    """
    try:
        response = client.prompts.list_versions(prompt=prompt_name, space=space)
    except Exception as exc:  # noqa: BLE001 - expected on the very first run
        console.print(f"  [dim]no existing versions ({type(exc).__name__}: {exc})[/dim]")
        return []
    return list(getattr(response, "data", None) or getattr(response, "prompt_versions", []))


def find_version(versions: list, text: str) -> str:
    """Id of an existing version whose system prompt is already `text`.

    Prompt versions are immutable, and `create_version` never deduplicates, so
    without this check each re-run of this script publishes another byte-identical
    copy -- the history fills with noise and a real diff becomes hard to find.
    """
    from copilot.prompts import system_text

    for version in versions:
        if system_text(version).strip() == text.strip():
            return str(getattr(version, "id", ""))
    return ""


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

    published = list_versions(client, PROMPT_NAME, space)

    # ---- v1: create the prompt -------------------------------------------
    console.print("[bold]Publishing v1[/bold] (the flawed baseline)")
    v1_id = find_version(published, V1)
    if v1_id:
        console.print(f"  [dim]already published as {v1_id}; reusing[/dim]")
    else:
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
        except Exception as exc:  # noqa: BLE001 - prompt exists with other content
            console.print(f"  [yellow]could not create ({exc}); adding as a version[/yellow]")
            v1_id = version_id(
                client.prompts.create_version(
                    prompt=PROMPT_NAME,
                    space=space,
                    commit_message="v1 baseline: ungrounded and never escalates",
                    input_variable_format=InputVariableFormat.NONE,
                    provider=PROVIDER,
                    model=MODEL,
                    messages=[message(V1)],
                )
            )

    # ---- v2: new immutable version ---------------------------------------
    console.print("\n[bold]Publishing v2[/bold] (grounded, escalates, concise)")
    v2_id = find_version(published, V2)
    if v2_id:
        console.print(f"  [dim]already published as {v2_id}; reusing[/dim]")
    else:
        v2 = client.prompts.create_version(
            prompt=PROMPT_NAME,
            space=space,
            commit_message=(
                "v2: require grounding in retrieved docs, admit gaps, escalate when "
                "blocked, cap length. Not yet proven better -- see poc/08."
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

        # strict: a silent fall back to the local copy would print a success
        # here and then run the agent on a prompt that never came from Arize,
        # which is the one thing this step is supposed to prove.
        published = load_prompt("hub", settings=settings, strict=True)
        expected = V2 if promote == "v2" else V1
        console.print(f"  fetched {len(published)} chars from Prompt Hub")
        check = (
            "[green]matches the promoted version[/green]"
            if published.strip() == expected.strip()
            else f"[yellow]differs from local {promote} — the hub copy is authoritative[/yellow]"
        )
        console.print(f"  {check}")

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
            # The load is already proven above (strict fetch + text match), so
            # this is not a plumbing problem: the published prompt reached the
            # agent and the agent hallucinated anyway. The KB documents that
            # cancellation stops future charges and says nothing about
            # proration, so any "no, there are no prorated refunds" is inferred
            # from an adjacent policy -- the exact move v2's grounding section
            # forbids. Consistent with poc/08, where groundedness did not
            # improve under v2.
            console.print(
                "[yellow]The agent answered confidently anyway.[/yellow] The prompt did "
                f"load from Prompt Hub (verified above), so `{PRODUCTION}` is not the "
                "problem — the wording is. The docs cover cancellation but not "
                "proration, so this answer infers a refund policy from an adjacent "
                "one.\n"
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

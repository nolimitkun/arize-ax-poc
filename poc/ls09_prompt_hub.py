#!/usr/bin/env python
"""Step ls09 -- Improve: publish to LangSmith's Prompt Hub, tag, load at runtime.

The LangSmith mirror of poc/09. Same idea -- the prompt stops being a string
in the repository and becomes a versioned artifact with a moving `production`
pointer -- with the platform nouns swapped: Arize versions carry *labels*,
LangSmith commits carry *tags*. `copilot.prompts.load_prompt("ls-hub")` is the
consumer side.

Same dedupe discipline as 09: commits are immutable and pushing never
deduplicates, so each candidate text is first looked for among the existing
commits -- otherwise every re-run of this script would bury the history in
byte-identical copies.

Docs: https://docs.smith.langchain.com/prompt_engineering/how_to_guides/manage_prompts_programatically
"""

from __future__ import annotations

import typer

from _common import console, done, header, table
from _ls_common import look_at_ls, ls_client, require_langsmith

app = typer.Typer(add_completion=False)

PRODUCTION = "production"


def make_template(text: str):
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages([("system", text)])


def existing_commits(client, name: str) -> list[tuple[str, str, str]]:
    """(commit_id, commit_hash, system_text) for every commit, newest first."""
    from copilot.prompts import ls_system_text

    try:
        listed = list(client.list_prompt_commits(name, limit=50))
    except Exception:  # noqa: BLE001 - expected on the very first run
        return []
    out = []
    for commit in listed:
        commit_hash = str(getattr(commit, "commit_hash", ""))
        try:
            pulled = client.pull_prompt(f"{name}:{commit_hash}")
            text = ls_system_text(pulled)
        except Exception:  # noqa: BLE001 - a commit that won't pull can't be reused
            continue
        out.append((str(getattr(commit, "id", "") or commit_hash), commit_hash, text))
    return out


def find_commit(commits: list[tuple[str, str, str]], text: str) -> tuple[str, str] | None:
    for commit_id, commit_hash, existing in commits:
        if existing.strip() == text.strip():
            return commit_id, commit_hash
    return None


@app.command()
def main(
    promote: str = typer.Option("v2", help="Which local version to tag `production`"),
    verify: bool = typer.Option(True, help="Run one live turn against the published prompt"),
) -> None:
    settings = header(
        "ls09",
        "Improve: publish to LangSmith Prompt Hub, tag, and load at runtime",
        "manage_prompts_programatically · pull_prompt",
    )
    require_langsmith(settings, "the Prompt Hub")

    from copilot.prompts import PROMPT_NAME, V1, V2

    client = ls_client(settings)
    commits = existing_commits(client, PROMPT_NAME)

    hashes: dict[str, tuple[str, str]] = {}
    for label, text, message in (
        ("v1", V1, "v1 baseline: helpful, but ungrounded and never escalates"),
        (
            "v2",
            V2,
            "v2: require grounding in retrieved docs, admit gaps, escalate when "
            "blocked, cap length. Not yet proven better -- see poc/ls08.",
        ),
    ):
        console.print(f"[bold]Publishing {label}[/bold]")
        found = find_commit(commits, text)
        if found:
            console.print(f"  [dim]already published as {found[1][:12]}; reusing[/dim]")
            hashes[label] = found
            continue
        url = client.push_prompt(
            PROMPT_NAME,
            object=make_template(text),
            description="System prompt for the Nimbus support copilot.",
            commit_description=message,
        )
        commits = existing_commits(client, PROMPT_NAME)
        found = find_commit(commits, text)
        if not found:
            console.print(
                f"[red]Pushed {label} but cannot find its commit reading back.[/red] {url}"
            )
            raise SystemExit(1)
        console.print(f"  created commit [bold]{found[1][:12]}[/bold]")
        hashes[label] = found

    # ---- tag -------------------------------------------------------------
    target_id, target_hash = hashes["v2" if promote == "v2" else "v1"]
    # The SDK's public tagging surface is `push_prompt(commit_tags=...)`, which
    # only tags a *new* commit -- moving a tag onto an existing commit goes
    # through the same POST /repos/{owner}/{name}/tags the SDK itself uses
    # inside push_prompt. "-" is how LangSmith spells "this workspace".
    client._create_commit_tags(f"-/{PROMPT_NAME}", target_id, [PRODUCTION])
    console.print(f"\n[green]Tagged commit {target_hash[:12]} as `{PRODUCTION}`[/green]")

    table(
        f"{PROMPT_NAME} commits",
        ["commit", "system prompt starts with"],
        [[h[:12], t.strip().splitlines()[0][:60]] for _id, h, t in commits],
    )

    # ---- consume it ------------------------------------------------------
    if verify:
        console.print("\n[bold]Loading the published prompt at runtime[/bold]")
        from copilot.agent import run_turn
        from copilot.prompts import load_prompt
        from copilot.tracing import flush, init_tracing

        # strict: a silent fall back to the local copy would print a success
        # here and prove nothing -- same stance as 09.
        published = load_prompt("ls-hub", settings=settings, strict=True)
        expected = V2 if promote == "v2" else V1
        console.print(f"  fetched {len(published)} chars from the Prompt Hub")
        if published.strip() != expected.strip():
            console.print(
                f"[bold red]`{PRODUCTION}` does not serve local {promote}.[/bold red] "
                f"Fetched {len(published)} chars, expected {len(expected.strip())}.\n"
                "Something else moved the tag, or the update has not propagated. "
                "Re-run to re-promote, or check the Prompt Hub before trusting any "
                "result below.\n"
            )
            raise typer.Exit(1)
        console.print("  [green]matches the promoted version[/green]")

        init_tracing(settings)
        probe = "If I cancel mid-month, do I get a prorated refund for the unused days?"
        console.print(f"\n  probe: [dim]{probe}[/dim]")
        result = run_turn(
            probe,
            settings=settings,
            prompt_version="ls-hub",
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
                "That behaviour came from LangSmith's Prompt Hub, not from this "
                "repository — no code changed between v1 and now.\n"
            )
        else:
            console.print(
                "[yellow]The agent answered confidently anyway.[/yellow] The prompt did "
                f"load from the Prompt Hub (verified above), so `{PRODUCTION}` is not "
                "the problem — the wording is. The docs cover cancellation but not "
                "proration, so this answer infers a refund policy from an adjacent "
                "one.\n"
            )

    look_at_ls(
        f"Prompts → {PROMPT_NAME}: the commit history, with a diff between any two.",
        f"The `{PRODUCTION}` tag — move it back to the v1 commit and re-run this "
        "script to watch the agent regress. That is your rollback path.",
        "Open the prompt in the Playground and run it against the "
        "copilot-failures-ls dataset.",
    )
    done("poc/ls10_dashboards.py — charts over the feedback these steps produced")


if __name__ == "__main__":
    app()

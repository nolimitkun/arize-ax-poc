#!/usr/bin/env python
"""Step ls05 -- Evaluate: automation rules, so evaluation runs in-platform.

The LangSmith mirror of poc/05. Same goal -- grading that happens continuously
on the server, not in a script that someone must remember to run -- reached
through LangSmith's automation *rules*: a filter over incoming runs plus an
action, sampled at a rate.

Two rules, mirroring 05's two evaluators:

  1. a routing rule that sends every run tagged `failure:hallucination` into
     the ls06 review queue -- the human loop stops depending on someone
     remembering to seed the queue
  2. an online LLM-as-judge rule -- which, like 05's online code evaluator on
     an un-provisioned Arize plan, needs something this script cannot create:
     a model API key configured inside the LangSmith workspace. It is
     attempted for real and reported honestly if the workspace isn't set up.

The SDK has no rules surface, so both go through the REST endpoint the UI
itself uses (`/runs/rules`), via the SDK client's authenticated session.

Docs: https://docs.smith.langchain.com/observability/how_to_guides/rules
"""

from __future__ import annotations

import json
from importlib import import_module

import typer

from _common import console, done, header, table
from _ls_common import look_at_ls, ls_client, require_langsmith

anno = import_module("06_annotations")

app = typer.Typer(add_completion=False)


def rule_name(base: str, project: str) -> str:
    """Project-scoped rule names -- the same collision this repo keeps finding.

    Rules are workspace-level and named; a fixed name would make the second
    project's tour find the first project's rule, conclude "already set up",
    and leave the new project with no automation at all while printing success.
    """
    return f"{base} ({project})"


def list_rules(client) -> list[dict]:
    response = client.request_with_retries("GET", "/runs/rules")
    response.raise_for_status()
    return response.json()


def post_rule(client, payload: dict) -> tuple[dict | None, str]:
    """(created rule, "") or (None, error text).

    request_with_retries *raises* on 4xx (LangSmithError wrapping the body)
    rather than returning the response, so the error path is an exception
    path -- and for the online judge a 400 is the expected outcome on a
    workspace with no model secret configured, not a crash.
    """
    try:
        response = client.request_with_retries(
            "POST", "/runs/rules", request_kwargs={"json": payload}
        )
        return response.json(), ""
    except Exception as exc:  # noqa: BLE001 - the caller reports it
        return None, str(exc)


@app.command()
def main(
    sampling_rate: float = typer.Option(1.0, help="Fraction of matching runs the rules touch"),
) -> None:
    settings = header(
        "ls05",
        "Evaluate: automation rules + online judge",
        "rules · online evaluators",
    )
    require_langsmith(settings, "automation rules")

    client = ls_client(settings)
    project = client.read_project(project_name=settings.langsmith_project)
    existing = {r.get("display_name", ""): r for r in list_rules(client)}

    # ---- 1. Route flagged runs into the review queue ---------------------
    routing_name = rule_name("Route hallucinations to review", settings.langsmith_project)
    console.print(f"[bold]Rule 1:[/bold] {routing_name}")
    queue = next(
        iter(client.list_annotation_queues(name=anno.queue_name(settings.langsmith_project))),
        None,
    )
    if queue is None:
        console.print(
            "  [yellow]The ls06 review queue does not exist yet[/yellow] — run "
            "poc/ls06_annotations.py first; this rule needs a queue to route into."
        )
    elif routing_name in existing:
        console.print(f"  [dim]already exists ({existing[routing_name]['id']}); reusing[/dim]")
    else:
        payload = {
            "display_name": routing_name,
            "session_id": str(project.id),
            "is_enabled": True,
            "sampling_rate": sampling_rate,
            "filter": 'and(eq(name, "copilot.turn"), has(tags, "failure:hallucination"))',
            "add_to_annotation_queue_id": str(queue.id),
        }
        rule, error = post_rule(client, payload)
        if rule is None:
            console.print(f"  [red]Rule not created:[/red] {error[:300]}")
        else:
            console.print(f"  [green]Created[/green] ({rule.get('id', '?')})")

    # ---- 2. Online LLM-as-judge ------------------------------------------
    judge_name = rule_name("Groundedness monitor", settings.langsmith_project)
    console.print(f"\n[bold]Rule 2:[/bold] {judge_name}")
    from copilot.evals import GROUNDEDNESS_TEMPLATE

    # The online judge grades from the run's own fields; LangSmith mustache-
    # substitutes input/output into the prompt server-side. Same limitation
    # 06b documents for Arize's hosted judge: no retrieved-context placeholder,
    # so it grades blind next to the offline judge.
    online_prompt = (
        GROUNDEDNESS_TEMPLATE.replace("{question}", "{{input.input}}")
        .replace("{answer}", "{{output.text}}")
        .replace("{retrieved_context}", "(not available to the online judge)")
    )
    judge_payload = {
        "display_name": judge_name,
        "session_id": str(project.id),
        "is_enabled": True,
        "sampling_rate": sampling_rate,
        "filter": 'eq(name, "copilot.turn")',
        "evaluators": [
            {
                "structured": {
                    "enabled": True,
                    "prompt": [
                        ["system", "You are grading a support assistant's answer."],
                        ["human", online_prompt],
                    ],
                    "schema": {
                        "title": "groundedness",
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "enum": ["grounded", "hallucinated"],
                            },
                        },
                        "required": ["label"],
                    },
                    "model": {},
                }
            }
        ],
    }
    if judge_name in existing:
        console.print(f"  [dim]already exists ({existing[judge_name]['id']}); reusing[/dim]")
    else:
        rule, error = post_rule(client, judge_payload)
        if rule is None:
            # The expected failure on a fresh workspace: online evaluators run
            # server-side on a model bound to a workspace secret, which no API
            # call can supply -- the server rejects an evaluator with no model
            # instance. Mirror of 05's plan-gated code evaluator: report it
            # plainly, don't pretend.
            console.print(f"  [yellow]Online judge not created.[/yellow] {error[:300]}")
            console.print(
                "\n  [dim]Online evaluators need a model secret configured in the "
                "workspace (Settings → Secrets) and an evaluator built with it, "
                "which is what the UI rule-builder produces. To finish by hand: "
                "Tracing project → Rules → Add Rule → Apply evaluator. The judge "
                "prompt this script would use:[/dim]"
            )
            console.print(f"  [dim]{json.dumps(judge_payload)[:400]}…[/dim]")
        else:
            console.print(f"  [green]Created[/green] ({rule.get('id', '?')})")

    # ---- verify by listing back ------------------------------------------
    rules = list_rules(client)
    ours = [r for r in rules if f"({settings.langsmith_project})" in r.get("display_name", "")]
    table(
        "Rules on this workspace (project-scoped)",
        ["rule", "enabled", "sampling"],
        [
            [r.get("display_name", "?"), r.get("is_enabled"), r.get("sampling_rate")]
            for r in ours
        ]
        or [["(none)", "-", "-"]],
    )
    if not ours:
        console.print(
            "[yellow]No rules exist for this project.[/yellow] Nothing is being "
            "evaluated or routed automatically — the sections above say why."
        )

    look_at_ls(
        "Tracing project → Rules: the routing rule, its filter, and its run log.",
        "Rules run on *arriving* traffic — re-run poc/01_trace.py and watch newly "
        "flagged turns appear in the review queue without ls06 seeding them.",
        "The rule's log shows every run it matched and what it did — the audit "
        "trail 05's online tasks keep in Arize.",
    )
    done(
        "poc/ls06_annotations.py — the queue these rules feed",
        "poc/ls07_dataset.py — turn the failures into a dataset",
    )


if __name__ == "__main__":
    app()

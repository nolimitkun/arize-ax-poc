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
  2. an online LLM-as-judge rule, which needs a model API key configured in
     the workspace itself (Settings -> Secrets); the rule's model reads it by
     *reference*, so no key passes through this script. Without that secret
     the step reports the gap instead of creating a rule that would fail on
     every run it touches.

What the online judge does NOT grade is groundedness, and the reason is worth
the detour. A rule's prompt binds only the matched run's own fields -- probed
live by swapping the evaluator for an echo: `{{input.input}}` and
`{{output.text}}` come back filled, `{{metadata.*}}` and `{{extra.metadata.*}}`
come back empty. The retrieved documentation lives in *child* retriever runs,
so it cannot reach a rule evaluating the parent turn. An online judge handed
the groundedness rubric with no documentation calls everything hallucinated --
verified, 6 of 6 -- and it would be reporting a number that means nothing.

A second rubric -- "does the answer assert a specific Nimbus policy as fact?"
-- is answerable blind, and was tried: it fired on 6 of 6 turns, because a
support copilot states specifics almost every time. Correctly wired, no
signal.

What is left is the one failure mode fully visible in the question and the
answer: a user who is blocked or angry, and a reply that never offers a human.
That is what this rule grades, under its own key. Note it is NOT the offline
`escalation_appropriate`, which decides from the fixture's expected tools --
same concern, different definition, so a separate key. Averaging two
definitions in one column is how a metric starts lying. Same asymmetry 06b
documents for Arize's hosted judge; ls04's offline judge, which does see the
documentation, stays the ground truth for groundedness.

Checked against known cases before being trusted, because "flagged nothing" and
"cannot flag anything" look identical from the outside: a real
missing_escalation turn from ls03 scores true, the same question answered with
an escalation scores false, and a calm informational turn scores false. On the
fresh traffic it then flagged 0 of 15 -- the copilot did escalate where it
mattered, which is a true negative rather than a silent judge.

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


# The workspace secret the online judge's model reads its key from. Rules run
# server-side: no local .env is visible to them, so the key must exist in
# LangSmith itself (Settings -> Secrets) under exactly this name.
JUDGE_SECRET = "DEEPSEEK_API_KEY"
JUDGE_MODEL = "deepseek-v4-pro"


def workspace_secrets(client) -> set[str]:
    """Names (never values) of the secrets configured in the workspace."""
    try:
        response = client.request_with_retries("GET", "/workspaces/current/secrets")
        return {str(row.get("key", "")) for row in response.json()}
    except Exception:  # noqa: BLE001 - absence of the listing is not absence of secrets
        return set()


def judge_model_spec() -> dict:
    """The judge's model as a serialized LangChain Runnable.

    This is what the `model` field actually is -- the server reconstructs the
    object and invokes it. An empty dict is the 400 ("Input should be an
    instance of Runnable") this step used to degrade on. Two details matter:
    the API key is a *secret reference* resolved server-side, and thinking
    must be disabled -- structured output forces tool_choice, which DeepSeek
    rejects while thinking.
    """
    return {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain_deepseek", "chat_models", "ChatDeepSeek"],
        "kwargs": {
            "model": JUDGE_MODEL,
            "extra_body": {"thinking": {"type": "disabled"}},
            "api_key": {"lc": 1, "type": "secret", "id": [JUDGE_SECRET]},
        },
    }


def post_rule(client, payload: dict, rule_id: str = "") -> tuple[dict | None, str]:
    """(rule, "") or (None, error text). PATCHes `rule_id` when given.

    request_with_retries *raises* on 4xx (LangSmithError wrapping the body)
    rather than returning the response, so the error path is an exception
    path -- and for the online judge a 400 is the expected outcome on a
    workspace with no model secret configured, not a crash.

    The update path matters as much as the create: a rule left over from an
    earlier version of this script keeps grading with whatever config it was
    born with, and "already exists, reusing" would report success while the
    live rule does the old, wrong thing.
    """
    method, path = ("PATCH", f"/runs/rules/{rule_id}") if rule_id else ("POST", "/runs/rules")
    try:
        response = client.request_with_retries(method, path, request_kwargs={"json": payload})
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
    # An earlier version of this step created a "Groundedness monitor" rule,
    # before the echo probe showed an online judge cannot see the retrieved
    # documentation. Renaming the rule does not retire the old one: it keeps
    # running, on every arriving turn, writing a verdict this repo no longer
    # stands behind. Deleting it is part of the change.
    for retired in (
        rule_name("Groundedness monitor", settings.langsmith_project),
        rule_name("Unsupported-claim monitor", settings.langsmith_project),
    ):
        if retired not in existing:
            continue
        try:
            client.request_with_retries("DELETE", f"/runs/rules/{existing[retired]['id']}")
            console.print(f"[dim]Retired the old {retired!r} rule.[/dim]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Could not delete {retired!r}: {exc}[/yellow]")

    judge_name = rule_name("Missed-escalation monitor", settings.langsmith_project)
    console.print(f"\n[bold]Rule 2:[/bold] {judge_name}")

    # Only the matched run's own input/output bind here (see this step's
    # header), so the rubric asks a question answerable from those two alone.
    online_prompt = """\
You are monitoring a customer-support assistant. You can see the user's
question and the assistant's answer, and nothing else.

[Question]
{{input.input}}

[Assistant answer]
{{output.text}}

Report `escalation_missed` as true only when BOTH hold:
  - the user is blocked, angry, or reporting something broken in production
    (repeated failures, lost data, an outage, an unanswered earlier request)
  - the answer never offers a human: no ticket, no support contact, no
    handover, no promise that someone will follow up

Report false in every other case -- including a calm informational question
answered informationally, and any answer that does offer a human.
"""
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
                        ["system", "You are monitoring a support assistant's answers."],
                        ["human", online_prompt],
                    ],
                    # Each *property* of the schema becomes a feedback key, and
                    # only a numeric/boolean one carries a score -- a string
                    # property lands as a value with score=None, which no chart
                    # can average. Hence a boolean named for the key we want,
                    # which LangSmith stores as 1.0/0.0.
                    "schema": {
                        "title": "escalation_missed",
                        "type": "object",
                        "properties": {
                            "escalation_missed": {
                                "type": "boolean",
                                "description": (
                                    "true if the user was blocked or angry and the answer "
                                    "never offered a human; false otherwise."
                                ),
                            },
                        },
                        "required": ["escalation_missed"],
                    },
                    "model": judge_model_spec(),
                }
            }
        ],
    }
    if JUDGE_SECRET not in workspace_secrets(client):
        # The expected gap on a fresh workspace: the rule's model reads its
        # key from a workspace secret, which no API call can supply. Mirror of
        # 05's plan-gated code evaluator: report it plainly, don't pretend.
        console.print(
            f"  [yellow]Online judge not created:[/yellow] the workspace has no "
            f"[bold]{JUDGE_SECRET}[/bold] secret, so the rule's model would fail "
            "on every run it grades."
        )
        console.print(
            "\n  [dim]Add it under Settings → Secrets (the value never passes "
            "through this script), then re-run this step. The payload it will "
            "post:[/dim]"
        )
        console.print(f"  [dim]{json.dumps(judge_payload)[:400]}…[/dim]")
    else:
        rule_id = existing[judge_name]["id"] if judge_name in existing else ""
        if rule_id:
            console.print(f"  [dim]exists ({rule_id}); updating it to this config[/dim]")
        rule, error = post_rule(client, judge_payload, rule_id)
        if rule is None:
            verb = "updated" if rule_id else "created"
            console.print(f"  [yellow]Online judge not {verb}.[/yellow] {error[:300]}")
            console.print(f"  [dim]{json.dumps(judge_payload)[:400]}…[/dim]")
        else:
            verb = "Updated" if rule_id else "Created"
            console.print(f"  [green]{verb}[/green] ({rule.get('id', rule_id or '?')})")

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

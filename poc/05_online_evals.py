#!/usr/bin/env python
"""Step 05 -- Evaluate: move evaluation into the platform, running continuously.

Step 04 evaluated a batch by hand. That doesn't scale to production. Here the
same two judgements become durable platform objects:

  evaluator (versioned template or code)  --  what the judgement is
  task      (project + filter + cadence)  --  where and how often it runs

Once created, AX evaluates new spans as they arrive, with no code in the loop,
and joins results back as `eval.<name>.*` attributes.

Docs: https://arize.com/docs/ax/concepts/evaluators/online-llm-as-judge
      https://arize.com/docs/ax/concepts/evaluators/online-code-evaluators
      https://arize.com/docs/ax/evaluate/create-evaluators
"""

from __future__ import annotations

import typer

from _common import arize_client, console, done, header, look_at, table

app = typer.Typer(add_completion=False)

GROUNDEDNESS_TEMPLATE = """\
You are grading a Nimbus customer-support answer for GROUNDEDNESS.

The assistant may only assert things supported by the documentation it
retrieved. When the documentation does not cover the question, the correct
behaviour is to say so -- not to state a plausible-sounding policy.

[Question]
{input.value}

[Assistant answer]
{output.value}

Reply "grounded" if every factual claim about Nimbus is supported by retrieved
documentation, or the assistant correctly said the docs don't cover it.

Reply "hallucinated" if it stated any specific policy, number, timeframe or
entitlement not present in the documentation -- even a plausible one. An
invented refund window is hallucination.
"""

# Runs inside AX against each matching span. Must define `evaluate`.
ESCALATION_CODE = '''
def evaluate(tool_calls, output):
    """Flag turns that should have escalated to a human but didn't."""
    calls = tool_calls or ""
    answer = (output or "").lower()

    distress = (
        "blocked", "urgent", "unacceptable", "outage", "leaked",
        "third time", "nobody has answered", "escalate", "right now",
    )
    looks_urgent = any(marker in answer for marker in distress)
    escalated = "escalate_ticket" in calls

    if looks_urgent and not escalated:
        return {
            "label": "missed",
            "score": 0.0,
            "explanation": "Distress signals present but escalate_ticket was not called.",
        }
    return {
        "label": "ok",
        "score": 1.0,
        "explanation": "Escalation behaviour looks appropriate for this turn.",
    }
'''


INTEGRATION_NAME = "deepseek-poc"


def evaluator_id(obj) -> str:
    return str(getattr(obj, "id", None) or getattr(getattr(obj, "evaluator", None), "id", ""))


def upsert_evaluator(client, space: str, name: str, create, new_version) -> tuple[str, str]:
    """Create an evaluator, or add a version to the one that's already there.

    Evaluators are versioned, so "already exists" is not an error condition --
    it's the second run. Without this, any partial failure downstream (a task
    that won't create, say) leaves the evaluator behind and every retry 409s,
    which is a miserable way to re-run a tour. Returns (id, what_happened).
    """
    from arize._generated.api_client.exceptions import ConflictException

    try:
        return evaluator_id(create()), "created"
    except ConflictException:
        new_version()
        existing = client.evaluators.get(evaluator=name, space=space)
        return evaluator_id(existing), "new version"


def find_task(client, space: str, name: str):
    listed = client.tasks.list(space=space, limit=100)
    items = getattr(listed, "data", None) or getattr(listed, "tasks", [])
    return next((t for t in items if getattr(t, "name", None) == name), None)


def upsert_task(
    client,
    space: str,
    name: str,
    create,
    *,
    desired: dict,
) -> tuple[str, str]:
    """Reuse an existing task of this name, bringing its config into line.

    Unlike evaluators, task names are *not* unique and creation never returns
    409 -- so this has to look before it leaps. Catching a conflict would be
    dead code, and every re-run would silently add another continuous task
    grading the same spans again, multiplying judge cost with no visible sign
    beyond a slowly growing Tasks list.

    Reusing blindly has the mirror-image problem: re-running with
    --sampling-rate 0.1 would leave the stored task at 1.0 while the closing
    summary reported 0.1. So compare what we asked for against what is stored,
    and update the drifted fields.
    """
    existing = find_task(client, space, name)
    if existing is None:
        return str(create().id), "created"

    task_id = str(getattr(existing, "id", ""))
    drift = {
        field: value
        for field, value in desired.items()
        if getattr(existing, field, None) != value
    }
    if not drift:
        return task_id, "reused"

    client.tasks.update(task=task_id, space=space, **drift)
    return task_id, "updated " + ", ".join(sorted(drift))


def create_deepseek_integration(client, settings) -> str | None:
    """Register DeepSeek in the Arize space as a custom OpenAI-compatible provider.

    Arize has no first-class DeepSeek provider, but `CUSTOM` takes a base URL
    and an API key, which is all an OpenAI-protocol endpoint needs.

    This uploads your DeepSeek key to Arize -- it has to, because the judge runs
    on their infrastructure, not here. That's why it sits behind a flag.
    """
    from arize.ai_integrations.types import AiIntegrationProvider

    try:
        integration = client.ai_integrations.create(
            name=INTEGRATION_NAME,
            provider=AiIntegrationProvider.CUSTOM,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model_names=["deepseek-v4-pro", "deepseek-v4-flash"],
            function_calling_enabled=True,
        )
        integration_id = getattr(integration, "id", None)
        console.print(f"[green]Created AI integration[/green] {INTEGRATION_NAME} ({integration_id})")
        console.print(
            f"[dim]Add ARIZE_AI_INTEGRATION_ID={integration_id} to .env to reuse it.[/dim]"
        )
        return str(integration_id) if integration_id else None
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Could not create the integration: {exc}[/yellow]")
        return None


def resolve_integration(client, settings) -> str | None:
    """Find a model provider configured *inside* Arize.

    Online LLM judges run on AX's infrastructure, so they need a provider
    registered in the space -- your local DEEPSEEK_API_KEY is not visible to
    them.
    """
    if settings.arize_ai_integration_id:
        return settings.arize_ai_integration_id
    try:
        found = client.ai_integrations.list(space=settings.arize_space_name, limit=50)
        integrations = getattr(found, "data", None) or getattr(found, "ai_integrations", [])
        if integrations:
            table(
                "AI integrations available in this space",
                ["id", "name", "provider"],
                [
                    [
                        getattr(i, "id", "?"),
                        getattr(i, "name", "?"),
                        str(getattr(i, "provider", "?")),
                    ]
                    for i in integrations
                ],
            )
            return getattr(integrations[0], "id", None)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Could not list AI integrations: {exc}[/yellow]")
    return None


def summarise(created: list[tuple[str, str, str]], sampling_rate: float) -> None:
    """Closing report. Shared, because the code-evaluator path may bail early."""
    table(
        "Created",
        ["kind", "name", "evaluator id"],
        [[k, n, i] for k, n, i in created],
    )
    console.print(
        f"\n[bold]These are continuous[/bold] at a sampling rate of {sampling_rate:g}. "
        "Re-run poc/01_trace.py to send fresh traffic; within a few minutes the "
        "new spans carry eval results with no code involved.\n"
    )
    look_at(
        "Eval Hub → the evaluators, with their versions and templates.",
        "Tasks → cadence, sampling rate and filter for each monitor.",
        "Traces (after new traffic) → `eval.Groundedness.label` appearing automatically.",
        "Edit the template in the UI, save a version, and note that the task picks it up.",
    )
    done("poc/06_annotations.py — human review, and judge-vs-human agreement")


@app.command()
def main(
    model_name: str = typer.Option("deepseek-v4-pro", help="Model the online judge uses"),
    sampling_rate: float = typer.Option(1.0, help="Fraction of spans to evaluate (0-1)"),
    skip_llm_judge: bool = typer.Option(False, help="Create only the code evaluator"),
    create_integration: bool = typer.Option(
        False,
        help="Register DeepSeek in the Arize space (uploads DEEPSEEK_API_KEY to Arize)",
    ),
) -> None:
    settings = header(
        "05",
        "Evaluate: online evaluators + continuous tasks",
        "online-llm-as-judge · online-code-evaluators · create-evaluators",
    )

    from arize.evaluators.types import (
        CustomCodeConfig,
        EvaluatorLlmConfig,
        OptimizationDirection,
        TemplateConfig,
    )
    from arize.tasks.types import TaskEvaluatorInput, TaskType

    client = arize_client(settings)
    space = settings.arize_space_name
    created: list[tuple[str, str, str]] = []

    # ---- 1. Online LLM-as-a-judge ---------------------------------------
    if not skip_llm_judge:
        integration_id = resolve_integration(client, settings)
        if not integration_id and create_integration:
            integration_id = create_deepseek_integration(client, settings)
        if not integration_id:
            console.print(
                "\n[yellow]No AI integration found in this space.[/yellow]\n"
                "Online LLM judges run on Arize's side and need a provider "
                "configured there — your local DEEPSEEK_API_KEY isn't visible to "
                "them. Either:\n"
                "  • re-run with --create-integration (registers DeepSeek as a "
                "CUSTOM OpenAI-compatible provider, uploading your key to Arize), or\n"
                "  • Arize → Space Settings → Integrations → add a Custom provider\n"
                f"      base URL {settings.deepseek_base_url}, models deepseek-v4-pro / "
                "deepseek-v4-flash\n"
                "  then put its ID in ARIZE_AI_INTEGRATION_ID.\n"
                "[dim]Skipping the LLM judge; the code evaluator below still works.[/dim]\n"
            )
        else:
            console.print(f"Using AI integration [bold]{integration_id}[/bold]")
            groundedness_config = TemplateConfig(
                name="Groundedness",
                template=GROUNDEDNESS_TEMPLATE,
                classification_choices={"grounded": 1, "hallucinated": 0},
                direction=OptimizationDirection.MAXIMIZE,
                include_explanations=True,
                # Both of AX's structured-output mechanisms are unusable
                # against DeepSeek V4, for the same reasons poc/04 hits
                # locally: `response_format: json_schema` is unsupported, and a
                # forced tool_choice is rejected while thinking mode is on (it
                # is on by default, and there is no reliable seam to disable it
                # from here -- InvocationParams has no `thinking` field).
                # Turning both off makes the judge emit a plain-text label,
                # which works either way.
                use_function_calling_if_available=False,
                use_structured_output=False,
                llm_config=EvaluatorLlmConfig(
                    ai_integration_id=integration_id,
                    model_name=model_name,
                    invocation_parameters={},
                    provider_parameters={},
                ),
            )
            eval_id, how = upsert_evaluator(
                client,
                space,
                "Groundedness",
                lambda: client.evaluators.create_template_evaluator(
                    name="Groundedness",
                    space=space,
                    commit_message="Flag support answers that invent undocumented policy",
                    description="Catches confident answers on topics the KB doesn't cover.",
                    template_config=groundedness_config,
                ),
                lambda: client.evaluators.create_template_version(
                    evaluator="Groundedness",
                    space=space,
                    commit_message="Text-mode labels: DeepSeek V4 rejects json_schema "
                    "and forced tool_choice under thinking mode",
                    template_config=groundedness_config,
                ),
            )
            created.append(("template", "Groundedness", eval_id))
            console.print(f"[green]Template evaluator {how}[/green] Groundedness ({eval_id})")

            task_id, how = upsert_task(
                client,
                space,
                "Groundedness monitor",
                lambda: client.tasks.create_evaluation_task(
                    name="Groundedness monitor",
                    task_type=TaskType.TEMPLATE_EVALUATION,
                    project=settings.arize_project_name,
                    # space is required for the project *name* to resolve.
                    space=space,
                    evaluators=[
                        TaskEvaluatorInput(
                            evaluator_id=eval_id,
                            column_mappings={
                                "input.value": "attributes.input.value",
                                "output.value": "attributes.output.value",
                            },
                        )
                    ],
                    is_continuous=True,
                    sampling_rate=sampling_rate,
                    # Only grade the agent-level span, not every child span.
                    query_filter="name = 'copilot.turn'",
                ),
                desired={
                    "is_continuous": True,
                    "sampling_rate": sampling_rate,
                    "query_filter": "name = 'copilot.turn'",
                },
            )
            console.print(f"[green]Task {how}[/green] Groundedness monitor ({task_id})\n")

    # ---- 2. Online code evaluator ---------------------------------------
    #
    # Online *code* evaluators are a paid entitlement. On an account without
    # them the API returns 400 "Custom code evals are not available for your
    # account", which is a plan boundary rather than a broken script -- so say
    # so and let the rest of the tour proceed. The same logic already ran
    # locally in poc/04, so nothing is left undemonstrated; only the
    # in-platform continuous version is unavailable.
    escalation_config = CustomCodeConfig(
        type="CUSTOM",
        name="EscalationAppropriate",
        code=ESCALATION_CODE,
        variables=["tool_calls", "output"],
    )
    try:
        code_id, how = upsert_evaluator(
            client,
            space,
            "EscalationAppropriate",
            lambda: client.evaluators.create_code_evaluator(
                name="EscalationAppropriate",
                space=space,
                commit_message="Flag distressed turns that never escalated",
                description="Deterministic check on the agent's escalation trajectory.",
                code_config=escalation_config,
            ),
            lambda: client.evaluators.create_code_version(
                evaluator="EscalationAppropriate",
                space=space,
                commit_message="Refresh escalation trajectory check",
                code_config=escalation_config,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        if "not available for your account" not in str(exc):
            raise
        console.print(
            "\n[yellow]Online code evaluators are not enabled on this account.[/yellow]\n"
            "  That's a plan entitlement, not a failure of this script. The same\n"
            "  escalation check already ran locally in poc/04 and its results are\n"
            "  on your spans; what's unavailable is running it continuously\n"
            "  inside AX. The online LLM judge above is unaffected.\n"
        )
        summarise(created, sampling_rate)
        return

    created.append(("code", "EscalationAppropriate", code_id))
    console.print(f"[green]Code evaluator {how}[/green] EscalationAppropriate ({code_id})")

    code_task_id, how = upsert_task(
        client,
        space,
        "Escalation monitor",
        lambda: client.tasks.create_evaluation_task(
            name="Escalation monitor",
            task_type=TaskType.CODE_EVALUATION,
            project=settings.arize_project_name,
            # space is required for the project *name* to resolve to an id.
            space=space,
            evaluators=[
                TaskEvaluatorInput(
                    evaluator_id=code_id,
                    column_mappings={
                        "tool_calls": "attributes.copilot.tool_calls",
                        "output": "attributes.output.value",
                    },
                )
            ],
            is_continuous=True,
            sampling_rate=sampling_rate,
            query_filter="name = 'copilot.turn'",
        ),
        desired={
            "is_continuous": True,
            "sampling_rate": sampling_rate,
            "query_filter": "name = 'copilot.turn'",
        },
    )
    console.print(f"[green]Task {how}[/green] Escalation monitor ({code_task_id})")

    summarise(created, sampling_rate)


if __name__ == "__main__":
    app()

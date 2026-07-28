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
            evaluator = client.evaluators.create_template_evaluator(
                name="Groundedness",
                space=space,
                commit_message="Flag support answers that invent undocumented policy",
                description="Catches confident answers on topics the KB doesn't cover.",
                template_config=TemplateConfig(
                    name="Groundedness",
                    template=GROUNDEDNESS_TEMPLATE,
                    classification_choices={"grounded": 1, "hallucinated": 0},
                    direction=OptimizationDirection.MAXIMIZE,
                    include_explanations=True,
                    use_function_calling_if_available=True,
                    llm_config=EvaluatorLlmConfig(
                        ai_integration_id=integration_id,
                        model_name=model_name,
                        invocation_parameters={},
                        provider_parameters={},
                    ),
                ),
            )
            eval_id = getattr(evaluator, "id", None) or getattr(evaluator.evaluator, "id")
            created.append(("template", "Groundedness", str(eval_id)))
            console.print(f"[green]Created template evaluator[/green] Groundedness ({eval_id})")

            task = client.tasks.create_evaluation_task(
                name="Groundedness monitor",
                task_type=TaskType.TEMPLATE_EVALUATION,
                project=settings.arize_project_name,
                evaluators=[
                    TaskEvaluatorInput(
                        evaluator_id=str(eval_id),
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
            )
            console.print(f"[green]Created task[/green] Groundedness monitor ({task.id})\n")

    # ---- 2. Online code evaluator ---------------------------------------
    code_evaluator = client.evaluators.create_code_evaluator(
        name="EscalationAppropriate",
        space=space,
        commit_message="Flag distressed turns that never escalated",
        description="Deterministic check on the agent's escalation trajectory.",
        code_config=CustomCodeConfig(
            type="custom",
            name="EscalationAppropriate",
            code=ESCALATION_CODE,
            variables=["tool_calls", "output"],
        ),
    )
    code_id = getattr(code_evaluator, "id", None) or getattr(code_evaluator.evaluator, "id")
    created.append(("code", "EscalationAppropriate", str(code_id)))
    console.print(f"[green]Created code evaluator[/green] EscalationAppropriate ({code_id})")

    code_task = client.tasks.create_evaluation_task(
        name="Escalation monitor",
        task_type=TaskType.CODE_EVALUATION,
        project=settings.arize_project_name,
        evaluators=[
            TaskEvaluatorInput(
                evaluator_id=str(code_id),
                column_mappings={
                    "tool_calls": "attributes.copilot.tool_calls",
                    "output": "attributes.output.value",
                },
            )
        ],
        is_continuous=True,
        sampling_rate=sampling_rate,
        query_filter="name = 'copilot.turn'",
    )
    console.print(f"[green]Created task[/green] Escalation monitor ({code_task.id})")

    table(
        "Created",
        ["kind", "name", "evaluator id"],
        [[k, n, i] for k, n, i in created],
    )

    console.print(
        "\n[bold]These are continuous.[/bold] Re-run poc/01_trace.py to send fresh "
        "traffic; within a few minutes the new spans carry eval results with no "
        "code involved.\n"
    )

    look_at(
        "Eval Hub → the two evaluators, with their versions and templates.",
        "Tasks → cadence, sampling rate and filter for each monitor.",
        "Traces (after new traffic) → `eval.Groundedness.label` appearing automatically.",
        "Edit the template in the UI, save a version, and note that the task picks it up.",
    )
    done("poc/06_annotations.py — human review, and judge-vs-human agreement")


if __name__ == "__main__":
    app()

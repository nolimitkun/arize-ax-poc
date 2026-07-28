#!/usr/bin/env python
"""Step 00 -- Self-check. Runs offline; no API keys, no network.

Verifies everything that can be verified without credentials, so that when you
do run the tour a failure means "credentials/platform", not "the repo is
broken". Checks:

  * the KB loads, and the deliberate refund gap is really a gap
  * the question fixture is well-formed
  * the code evaluators fire on the failure modes they're supposed to catch
  * v1 and v2 prompts differ in the ways the POC claims
  * eval / annotation dataframes match the Arize SDK's column patterns
"""

from __future__ import annotations

import sys

import pandas as pd

from _common import console, table

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        console.print(f"  [green]✓[/green] {name}")
    else:
        console.print(f"  [red]✗[/red] {name} [dim]{detail}[/dim]")
        FAILURES.append(name)


def check_knowledge_base() -> None:
    console.print("\n[bold]Knowledge base[/bold]")
    from copilot.kb import _index, search

    chunks, _, _, _ = _index()
    docs = {c.source for c in chunks}
    check(f"corpus loads ({len(chunks)} chunks, {len(docs)} docs)", len(chunks) > 40)

    # The gap is the whole point. Note what is NOT asserted: that retrieval
    # returns nothing. It returns weak, adjacent matches (a chunk mentioning
    # "key rotation policy" scores on the word "policy"), and that is desirable
    # -- plausible-but-irrelevant context is exactly what tempts a model into
    # inventing an answer. What must hold is that nothing retrieved actually
    # answers the question.
    corpus_text = " ".join(c.text.lower() for c in chunks)
    check(
        "no document mentions refunds at all",
        "refund" not in corpus_text,
        "a doc mentions refunds -- the hallucination test is compromised",
    )

    for query in ("What is your refund policy?", "Do I get a prorated refund if I cancel?"):
        hits = search(query)
        answering = [c.doc_id for c, _ in hits if "refund" in c.text.lower()]
        check(
            f"no chunk answers {query[:40]!r}",
            not answering,
            f"these would answer it: {answering}",
        )

    # ...but ordinary questions must retrieve well, or the agent fails for the
    # wrong reason and the POC measures nothing.
    for query, expect in [
        ("What scopes can an API key have?", "api-keys.md"),
        ("How many times do you retry a failed webhook?", "webhooks.md"),
        ("What is the uptime SLA on Business?", "status-and-incidents.md"),
        ("How do I fix schema_drift?", "troubleshooting-runs.md"),
    ]:
        hits = search(query)
        top = hits[0][0].source if hits else "(nothing)"
        check(f"retrieval: {query[:44]!r} → {expect}", top == expect, f"got {top}")


def check_questions() -> None:
    console.print("\n[bold]Question fixture[/bold]")
    from copilot.agent import load_questions

    questions = load_questions()
    check(f"questions.jsonl parses ({len(questions)} rows)", len(questions) >= 40)

    required = {"id", "question", "expected_behavior", "expected_tools", "topic", "failure_mode"}
    check("every row has the required fields", all(required <= set(q) for q in questions))
    check("ids are unique", len({q["id"] for q in questions}) == len(questions))

    modes = {q["failure_mode"] for q in questions if q["failure_mode"]}
    for mode in ("hallucination", "wrong_tool", "missing_escalation", "verbosity"):
        count = sum(1 for q in questions if q["failure_mode"] == mode)
        check(f"seeded failure mode `{mode}` present ({count} questions)", count >= 4)

    seeded = sum(1 for q in questions if q["failure_mode"])
    share = seeded / len(questions)
    check(
        f"seeded failures are {share:.0%} of the set (want 30-70%)",
        0.3 <= share <= 0.7,
    )


def check_code_evaluators() -> None:
    console.print("\n[bold]Code evaluators[/bold]")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import importlib

    offline = importlib.import_module("04_offline_evals")
    observe = importlib.import_module("03_query_spans")

    hallucinated = observe.check_ungrounded(
        "What is your refund policy?",
        "You have 30 days from the charge date to request a full refund.",
        [],
    )
    check("ungrounded check catches an invented refund window", hallucinated)

    hedged = observe.check_ungrounded(
        "What is your refund policy?",
        "Our documentation doesn't cover refunds, so I don't want to guess. "
        "I can escalate this to a human.",
        [],
    )
    check("ungrounded check accepts an honest refusal", not hedged)

    unrelated = observe.check_ungrounded(
        "What scopes can an API key have?", "read, write, run and admin.", ["api-keys.md#Scopes"]
    )
    check("ungrounded check ignores non-refund questions", not unrelated)

    check(
        "wrong-tool check fires when lookup_order is skipped",
        observe.check_wrong_tool(["lookup_order"], ["search_docs"]),
    )
    check(
        "wrong-tool check passes when lookup_order is used",
        not observe.check_wrong_tool(["lookup_order"], ["lookup_order"]),
    )
    check(
        "escalation check fires on a missed escalation",
        observe.check_missing_escalation(["escalate_ticket"], ["search_docs"]),
    )
    check("verbosity check fires past 250 words", observe.check_verbose("word " * 300))
    check("verbosity check passes a short answer", not observe.check_verbose("short answer"))

    # The same logic, as used against the eval dataframe in step 04.
    row = pd.Series({"expected_tools": "lookup_order", "tool_calls": "search_docs"})
    label, score, _ = offline.eval_tool_selection(row)
    check("eval_tool_selection labels a miss `incorrect`", label == "incorrect" and score == 0.0)

    row = pd.Series({"expected_tools": "escalate_ticket", "tool_calls": ""})
    label, score, _ = offline.eval_escalation(row)
    check("eval_escalation labels a miss `missed`", label == "missed" and score == 0.0)

    label, score, _ = offline.eval_conciseness(pd.Series({"answer_words": 400}))
    check("eval_conciseness labels 400 words `verbose`", label == "verbose" and score == 0.0)


def check_prompts_and_tools() -> None:
    console.print("\n[bold]Prompts and tool schemas[/bold]")
    from copilot.prompts import V1, V2
    from copilot.tools import tools_for

    check("v2 is substantially longer than v1", len(V2) > len(V1) * 3)
    for term in ("ground", "escalate", "150 words"):
        check(f"v2 addresses {term!r}", term.lower() in V2.lower())
    for term in ("ground", "escalate"):
        check(f"v1 does NOT address {term!r} (it's the baseline)", term.lower() not in V1.lower())

    from copilot.tools import description_of

    v1_tools, v2_tools = tools_for("v1"), tools_for("v2")
    check("both versions expose 3 tools", len(v1_tools) == len(v2_tools) == 3)
    check(
        "v2 tool descriptions are far more specific",
        sum(len(description_of(t)) for t in v2_tools)
        > sum(len(description_of(t)) for t in v1_tools) * 5,
    )
    # OpenAI function-calling shape, which is what DeepSeek's endpoint accepts.
    for tools, version in ((v1_tools, "v1"), (v2_tools, "v2")):
        ok = all(
            t.get("type") == "function"
            and {"name", "description", "parameters"} <= set(t.get("function", {}))
            and t["function"]["parameters"].get("type") == "object"
            for t in tools
        )
        check(f"{version} tool schemas are well-formed", ok)


def check_dataframe_contracts() -> None:
    """The highest-value offline check: the SDK validates these with regexes."""
    console.print("\n[bold]Arize dataframe contracts[/bold]")
    from arize.spans.columns import ANNOTATION_COLUMN_PATTERN, EVAL_COLUMN_PATTERN

    import re

    eval_re = re.compile(EVAL_COLUMN_PATTERN)
    annot_re = re.compile(ANNOTATION_COLUMN_PATTERN)

    from importlib import import_module

    offline = import_module("04_offline_evals")

    eval_names = [
        offline.GROUNDEDNESS,
        offline.TOOL_SELECTION,
        offline.ESCALATION,
        offline.CONCISENESS,
    ]
    for name in eval_names:
        cols = [f"eval.{name}.{suffix}" for suffix in ("label", "score", "explanation")]
        ok = all(eval_re.match(c) for c in cols)
        check(f"eval columns for `{name}` match the SDK pattern", ok, cols[0])

    annotations = import_module("06_annotations")
    annot_cols = [
        f"annotation.{annotations.ANNOTATION_NAME}.{s}"
        for s in ("label", "score", "text", "updated_by", "updated_at")
    ]
    check(
        f"annotation columns for `{annotations.ANNOTATION_NAME}` match the SDK pattern",
        all(annot_re.match(c) for c in annot_cols),
        annot_cols[0],
    )


def check_targets_ax_not_phoenix() -> None:
    """This POC exercises Arize AX, the hosted platform -- not Arize Phoenix.

    The two share OpenInference conventions and a lot of vocabulary, so it is
    easy for a project to drift from one to the other without anyone noticing.
    These checks pin the distinction down:

      * traces must leave via arize.otel, aimed at Arize's OTLP collector
      * every platform operation must go through ArizeClient
      * nothing may import phoenix.otel, phoenix.trace, or run a Phoenix server

    `phoenix.evals` is the deliberate exception. It is a standalone evaluation
    library, and the AX docs themselves prescribe it for code-based evals
    (arize.com/docs/ax/evaluate/run-evals-on-traces) -- scores are computed
    locally, then written to AX with client.spans.update_evaluations().
    """
    console.print("\n[bold]Targets AX, not Phoenix[/bold]")
    import os
    from pathlib import Path

    from arize.otel import Endpoint

    check(
        "trace exporter defaults to the Arize collector",
        Endpoint.ARIZE.value == "https://otlp.arize.com/v1",
        Endpoint.ARIZE.value,
    )
    # register() honours this before its default; a stale value would silently
    # send the whole tour somewhere other than AX.
    override = os.getenv("ARIZE_COLLECTOR_ENDPOINT")
    check(
        "no ARIZE_COLLECTOR_ENDPOINT override redirecting traces",
        not override or "arize.com" in override,
        f"set to {override!r}",
    )

    root = Path(__file__).resolve().parents[1]
    # This file is excluded: it names the banned tokens in order to look for them.
    sources = [
        p
        for p in (*(root / "src" / "copilot").glob("*.py"), *(root / "poc").glob("*.py"))
        if p.name != Path(__file__).name
    ]
    banned = ("phoenix.otel", "phoenix.trace", "phoenix.session", "px.launch_app")
    offenders = [
        f"{p.name}:{token}"
        for p in sources
        for token in banned
        if token in p.read_text(encoding="utf-8")
    ]
    check("no Phoenix tracing/server imports", not offenders, str(offenders))

    tracing_src = (root / "src" / "copilot" / "tracing.py").read_text(encoding="utf-8")
    check("tracing bootstrap registers via arize.otel", "from arize.otel import register" in tracing_src)


def check_sdk_surface() -> None:
    console.print("\n[bold]SDK surface[/bold]")
    try:
        from arize.client import ArizeClient

        client = ArizeClient(api_key="offline-selfcheck")
        for resource in (
            "datasets",
            "experiments",
            "evaluators",
            "tasks",
            "spans",
            "prompts",
            "annotation_configs",
            "annotation_queues",
            "ai_integrations",
            "projects",
        ):
            check(f"client.{resource} exists", hasattr(client, resource))
    except Exception as exc:  # noqa: BLE001
        check("arize SDK imports", False, str(exc))

    for module, symbol in [
        ("arize.experiments", "EvaluationResult"),
        ("arize.evaluators.types", "TemplateConfig"),
        ("arize.evaluators.types", "EvaluatorLlmConfig"),
        ("arize.evaluators.types", "CustomCodeConfig"),
        ("arize.tasks.types", "TaskType"),
        ("arize.tasks.types", "TaskEvaluatorInput"),
        ("arize.prompts.types", "LLMMessage"),
        ("arize.annotation_configs.types", "CategoricalAnnotationValue"),
        ("arize.otel", "register"),
        ("openinference.instrumentation.openai", "OpenAIInstrumentor"),
        ("arize.ai_integrations.types", "AiIntegrationProvider"),
        ("phoenix.evals", "create_classifier"),
    ]:
        try:
            mod = __import__(module, fromlist=[symbol])
            check(f"{module}.{symbol}", hasattr(mod, symbol))
        except Exception as exc:  # noqa: BLE001
            check(f"{module}.{symbol}", False, str(exc))


def main() -> None:
    console.print("\n[bold cyan]Arize AX POC — offline self-check[/bold cyan]")
    console.print("[dim]No credentials or network required.[/dim]")

    check_knowledge_base()
    check_questions()
    check_code_evaluators()
    check_prompts_and_tools()
    check_dataframe_contracts()
    check_targets_ax_not_phoenix()
    check_sdk_surface()

    console.print()
    if FAILURES:
        table("Failed checks", ["check"], [[f] for f in FAILURES])
        console.print(f"\n[bold red]{len(FAILURES)} check(s) failed.[/bold red]\n")
        raise SystemExit(1)
    console.print("[bold green]All checks passed.[/bold green]")
    console.print(
        "\nNext: copy .env.example to .env, fill in credentials, then\n"
        "  [bold]uv run python poc/01_trace.py[/bold]\n"
    )


if __name__ == "__main__":
    main()

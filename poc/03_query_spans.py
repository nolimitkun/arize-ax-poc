#!/usr/bin/env python
"""Step 03 -- Observe: export spans and find the failing traces.

This is the "find failures" step the Improve-Your-Agent guide opens with. It
pulls spans back out of AX with the SDK, reconstructs each agent turn's
trajectory, and applies the code checks that identify the seeded failure modes.

What it finds is then written *back* onto the spans as metadata, so the trace
view can be filtered by failure mode. An analysis that stays in a local
dataframe leaves the UI no better than it was.

The output feeds steps 04 (evals), 06 (annotation) and 07 (dataset).

Docs: https://arize.com/docs/ax/observe/tracing/view-and-manage-traces
      https://arize.com/docs/api-clients/python/version-8/client-resources/spans
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import typer

from _common import arize_client, console, done, header, look_at, save, table, window

app = typer.Typer(add_completion=False)


def find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Locate a column by exact name, then by suffix.

    Arize's export flattens span attributes into columns, and the exact prefix
    has varied across SDK versions. Matching on suffix keeps this working
    without pinning to one layout.
    """
    for name in candidates:
        if name in df.columns:
            return name
    for name in candidates:
        for col in df.columns:
            if col.endswith(name):
                return col
    return None


def as_list(value: Any) -> list[str]:
    """Span list-attributes come back as list, ndarray, JSON string, or NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                return [str(v) for v in json.loads(stripped)]
            except json.JSONDecodeError:
                return [stripped] if stripped else []
        return [stripped] if stripped else []
    try:
        return [str(v) for v in value]
    except TypeError:
        return [str(value)]


# --- the code checks that define "a failure" for this POC ------------------

REFUND_MARKERS = ("refund", "money-back", "money back", "prorated refund")
HEDGE_MARKERS = (
    "doesn't cover",
    "does not cover",
    "not documented",
    "isn't documented",
    "no documentation",
    "don't want to guess",
    "not in our documentation",
    "couldn't find",
    "could not find",
    "unable to find",
    "i don't have",
    "i do not have",
)


def check_ungrounded(question: str, answer: str, doc_ids: list[str]) -> bool:
    """Answered a refund question with a confident policy we never documented."""
    if not any(m in question.lower() for m in REFUND_MARKERS):
        return False
    if not answer.strip():
        return False
    return not any(h in answer.lower() for h in HEDGE_MARKERS)


def check_wrong_tool(expected: list[str], actual: list[str]) -> bool:
    """Needed lookup_order and didn't call it."""
    return "lookup_order" in expected and "lookup_order" not in actual


def check_missing_escalation(expected: list[str], actual: list[str]) -> bool:
    return "escalate_ticket" in expected and "escalate_ticket" not in actual


def check_verbose(answer: str, limit: int = 250) -> bool:
    return len(answer.split()) > limit


def enrich_spans(client, settings, turns: pd.DataFrame) -> None:
    """Write the failure classification back onto the spans it describes.

    Without this the verdicts live only in `.out/03_turns.parquet`, which means
    the one place you cannot answer "show me the hallucinations" is the trace
    view -- the place you go when you want to read one. `update_metadata` takes
    a JSON Merge Patch, and any `attributes.metadata.*` column is converted into
    one automatically, so the columns below become filterable span fields.

    Metadata rather than an eval: these are code checks over a local fixture,
    not a graded judgement, and mixing them into `eval.*` would put two
    different kinds of claim in the same namespace. Step 04 owns `eval.*`.
    """
    patch = pd.DataFrame(
        {
            "context.span_id": turns["span_id"].astype(str),
            # "none" rather than "" -- an empty string is indistinguishable from
            # an unset field in the UI filter, and "no failures" is a finding.
            "attributes.metadata.failure_mode": turns["failures"].replace("", "none"),
            "attributes.metadata.failure_count": turns["failures"].apply(
                lambda f: len([m for m in f.split(",") if m])
            ),
            "attributes.metadata.question_id": turns["question_id"].astype(str),
            "attributes.metadata.expected_behavior": turns["expected_behavior"].astype(str),
            "attributes.metadata.answer_words": turns["answer_words"].astype(int),
        }
    )
    client.spans.update_metadata(
        space_id=settings.arize_space_id,
        project_name=settings.arize_project_name,
        dataframe=patch,
    )
    console.print(
        f"[green]Tagged {len(patch)} spans[/green] with "
        "`metadata.failure_mode` — filterable in the trace view."
    )


@app.command()
def main(
    hours: int = typer.Option(24, help="How far back to export"),
    limit: int = typer.Option(0, help="Cap rows for a quick look (0 = no cap)"),
    skip_metadata: bool = typer.Option(
        False, help="Don't write the failure classification back onto the spans"
    ),
) -> None:
    settings = header(
        "03",
        "Observe: export spans, find the failing traces",
        "view-and-manage-traces · spans client resource",
    )

    client = arize_client(settings)
    start, end = window(hours)

    console.print(f"Exporting spans from the last {hours}h…")
    df = client.spans.export_to_df(
        space_id=settings.arize_space_id,
        project_name=settings.arize_project_name,
        start_time=start,
        end_time=end,
    )
    if df.empty:
        console.print(
            "[red]No spans returned.[/red] Run poc/01_trace.py first, and give "
            "ingestion a minute — spans are batched."
        )
        raise SystemExit(1)

    console.print(f"[green]{len(df)} spans[/green] across {df.shape[1]} columns.\n")

    kind_col = find_col(df, "attributes.openinference.span.kind", "openinference.span.kind")
    name_col = find_col(df, "name", "attributes.name")
    if kind_col:
        counts = df[kind_col].value_counts()
        table(
            "Spans by OpenInference kind",
            ["kind", "count"],
            [[k, v] for k, v in counts.items()],
        )

    # The AGENT span is the unit of analysis: one per user turn, carrying the
    # question, the final answer, and the trajectory we recorded in agent.py.
    agent_df = df[df[name_col] == "copilot.turn"].copy() if name_col else df.copy()
    if agent_df.empty:
        console.print("[red]No `copilot.turn` spans found.[/red] Was poc/01 run?")
        raise SystemExit(1)

    cols = {
        "span_id": find_col(agent_df, "context.span_id", "span_id"),
        "trace_id": find_col(agent_df, "context.trace_id", "trace_id"),
        "input": find_col(agent_df, "attributes.input.value", "input.value"),
        "output": find_col(agent_df, "attributes.output.value", "output.value"),
        "tools": find_col(agent_df, "attributes.copilot.tool_calls", "copilot.tool_calls"),
        "docs": find_col(
            agent_df, "attributes.copilot.retrieved_doc_ids", "copilot.retrieved_doc_ids"
        ),
        "version": find_col(
            agent_df, "attributes.copilot.prompt_version", "copilot.prompt_version"
        ),
        "session": find_col(agent_df, "attributes.session.id", "session.id"),
    }
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        console.print(f"[yellow]Columns not found in export: {missing}[/yellow]")
        console.print(f"[dim]Available: {sorted(agent_df.columns)[:40]}…[/dim]")

    # Expected behaviour lives in the local fixture, joined back by question text.
    from copilot.agent import load_questions

    expectations = {q["question"]: q for q in load_questions()}

    rows = []
    for _, span in agent_df.iterrows():
        question = str(span.get(cols["input"], "") or "")
        answer = str(span.get(cols["output"], "") or "")
        tools = as_list(span.get(cols["tools"])) if cols["tools"] else []
        docs = as_list(span.get(cols["docs"])) if cols["docs"] else []
        meta = expectations.get(question, {})
        expected_tools = meta.get("expected_tools", [])

        failures = []
        if check_ungrounded(question, answer, docs):
            failures.append("hallucination")
        if check_wrong_tool(expected_tools, tools):
            failures.append("wrong_tool")
        if check_missing_escalation(expected_tools, tools):
            failures.append("missing_escalation")
        if check_verbose(answer):
            failures.append("verbosity")

        rows.append(
            {
                "span_id": span.get(cols["span_id"], ""),
                "trace_id": span.get(cols["trace_id"], ""),
                "session_id": span.get(cols["session"], "") if cols["session"] else "",
                "question_id": meta.get("id", ""),
                "question": question,
                "answer": answer,
                "tool_calls": ",".join(tools),
                "retrieved_doc_ids": ",".join(docs),
                "expected_behavior": meta.get("expected_behavior", ""),
                "expected_tools": ",".join(expected_tools),
                "prompt_version": span.get(cols["version"], "") if cols["version"] else "",
                "answer_words": len(answer.split()),
                "failures": ",".join(failures),
                "is_failure": bool(failures),
            }
        )

    turns = pd.DataFrame(rows)
    if limit:
        turns = turns.head(limit)

    counts: dict[str, int] = {}
    for entry in turns["failures"]:
        for f in filter(None, entry.split(",")):
            counts[f] = counts.get(f, 0) + 1

    total = len(turns)
    failing = int(turns["is_failure"].sum())
    table(
        "Failure modes detected",
        ["mode", "turns", "% of turns"],
        [[m, c, f"{100 * c / total:.0f}%"] for m, c in sorted(counts.items(), key=lambda x: -x[1])]
        or [["(none)", 0, "0%"]],
    )
    console.print(
        f"\n[bold]{failing}/{total}[/bold] turns show at least one failure "
        f"([bold]{100 * failing / total:.0f}%[/bold]).\n"
    )

    if failing:
        console.print("[bold]Examples:[/bold]")
        for _, r in turns[turns["is_failure"]].head(3).iterrows():
            console.print(f"  [yellow]{r['failures']}[/yellow]  {r['question'][:70]}")
            console.print(f"    [dim]{r['answer'][:110].replace(chr(10), ' ')}…[/dim]")

    save("03_turns.parquet", turns)
    save("03_failures.parquet", turns[turns["is_failure"]].copy())

    if not skip_metadata:
        console.print("\nTagging the spans with what was found…")
        try:
            enrich_spans(client, settings, turns)
        except Exception as exc:  # noqa: BLE001 - the export above is the real work
            console.print(
                f"[yellow]Could not write span metadata: {type(exc).__name__}: {exc}[/yellow]"
            )

    look_at(
        "Traces → sort by latency, and open the slowest turn.",
        "Filter to the refund questions and read the answers — that is the "
        "hallucination this whole loop exists to fix.",
        "Compare an order question's trace against `expected_tools`: v1 reaches "
        "for search_docs instead of lookup_order.",
        "Filter `metadata.failure_mode = 'hallucination'` — the classification "
        "this step just computed is now a span field, so the failing turns are "
        "one filter away instead of one parquet file away.",
    )
    done(
        "poc/04_offline_evals.py — LLM-as-judge over these spans",
        "poc/07_dataset.py — turn the failures into a dataset",
    )


if __name__ == "__main__":
    app()

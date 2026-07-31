#!/usr/bin/env python
"""Step 02b -- Instrument: get spans into AX without OpenTelemetry.

Everything else in this tour reaches AX through the OTel exporter, which is the
right path for a live application. It is the wrong path for history: the
conversations your support team handled last quarter are rows in a warehouse,
not running processes you can instrument.

`client.spans.log()` is the other door. You hand it a dataframe shaped like
OpenInference spans and it ingests them directly -- no tracer, no collector, no
running agent. That is how you backfill a baseline to compare against, or bring
in a system you cannot instrument at all.

The synthetic history goes to its own project. Mixing backfilled rows into the
project step 03 analyses would quietly change every count that follows -- span
totals, failure rates, the dataset built from them -- and none of it would be
labelled as coming from a different source.

Docs: https://arize.com/docs/api-clients/python/version-8/client-resources/spans
      https://arize.com/docs/ax/observe/tracing/view-and-manage-traces
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import typer

from _common import arize_client, console, done, header, look_at, table

app = typer.Typer(add_completion=False)

# Stand-in for "what the warehouse has": question, answer, whether the legacy
# system got it right. Deliberately includes the same refund-policy failure the
# live agent has, so the backfill is comparable to what step 03 measures.
HISTORIC_TICKETS = [
    (
        "Can I get a refund if I cancel halfway through the year?",
        "Yes -- we refund the unused months on annual plans, prorated to the day.",
        0.0,
        "hallucinated",
    ),
    (
        "How do I add a second workspace to my account?",
        "Open Settings -> Workspaces and choose 'Create workspace'. Team and "
        "Business plans include unlimited workspaces.",
        1.0,
        "grounded",
    ),
    (
        "What's the retention period for deleted projects?",
        "Deleted projects stay recoverable for 30 days, then they're purged.",
        1.0,
        "grounded",
    ),
    (
        "Do you offer a discount for non-profits?",
        "We do -- non-profits get 40% off any plan, just email billing with your "
        "registration number.",
        0.0,
        "hallucinated",
    ),
    (
        "My exports keep timing out on large projects.",
        "Exports over 2GB run asynchronously; you'll get an email when the file "
        "is ready. If it's still failing, I've raised a ticket for engineering.",
        1.0,
        "grounded",
    ),
]

SPAN_KIND = "attributes.openinference.span.kind"


# Ids are derived from the ticket rather than random, so re-running replaces the
# same spans instead of adding a second copy of the history. A backfill is a
# statement about what happened; running it twice should not double it.
_NAMESPACE = uuid.UUID("6f9d5b1e-0c3a-4b7d-9e2f-1a8c4d6b0e57")


def stable_id(*parts: str, length: int = 32) -> str:
    """A deterministic hex id: 32 chars for a trace, 16 for a span."""
    return uuid.uuid5(_NAMESPACE, "|".join(parts)).hex[:length]


def build_history(tickets: list[tuple[str, str, float, str]], days_back: int):
    """A spans dataframe and its matching evals dataframe.

    Two spans per ticket -- an AGENT root and its LLM child -- because a flat
    list of root spans would demo ingestion without demonstrating that the
    parent/child structure survives it. `parent_id` is what AX rebuilds the
    trace tree from.
    """
    spans, evals = [], []
    base = datetime.now(timezone.utc) - timedelta(days=days_back)

    for i, (question, answer, score, label) in enumerate(tickets):
        trace = stable_id("trace", str(i))
        root = stable_id("root", str(i), length=16)
        child = stable_id("llm", str(i), length=16)
        start = base + timedelta(minutes=7 * i)
        llm_start = start + timedelta(milliseconds=120)
        llm_end = llm_start + timedelta(milliseconds=1500 + 90 * i)
        end = llm_end + timedelta(milliseconds=40)

        spans.append(
            {
                "context.trace_id": trace,
                "context.span_id": root,
                "parent_id": "",
                "name": "legacy.ticket",
                SPAN_KIND: "AGENT",
                "start_time": start,
                "end_time": end,
                "status_code": "OK",
                "attributes.input.value": question,
                "attributes.output.value": answer,
                "attributes.session.id": f"legacy-{i:03d}",
                "attributes.user.id": f"customer-{i:03d}",
                "attributes.metadata": {
                    "source": "warehouse-backfill",
                    "handled_by": "legacy-macro-system",
                },
            }
        )
        spans.append(
            {
                "context.trace_id": trace,
                "context.span_id": child,
                "parent_id": root,
                "name": "legacy.answer",
                SPAN_KIND: "LLM",
                "start_time": llm_start,
                "end_time": llm_end,
                "status_code": "OK",
                "attributes.input.value": question,
                "attributes.output.value": answer,
                "attributes.llm.model_name": "legacy-macro-v1",
                "attributes.llm.token_count.prompt": 180 + 12 * i,
                "attributes.llm.token_count.completion": len(answer.split()),
                "attributes.llm.token_count.total": 180 + 12 * i + len(answer.split()),
                "attributes.session.id": f"legacy-{i:03d}",
                "attributes.user.id": f"customer-{i:03d}",
            }
        )
        # The eval frame joins to spans on context.span_id, so a backfill can
        # carry its verdicts with it rather than needing a second grading pass
        # over data that was already graded.
        evals.append(
            {
                "context.span_id": root,
                "eval.groundedness.label": label,
                "eval.groundedness.score": score,
                "eval.groundedness.explanation": (
                    "Backfilled verdict from the legacy quality review."
                ),
            }
        )

    return pd.DataFrame(spans), pd.DataFrame(evals)


POLL_ATTEMPTS = 8
POLL_SECONDS = 30


def wait_for_spans(client, settings, target: str, days_back: int):
    """Poll until the logged spans are queryable, or give up honestly.

    HTTP 200 from `log()` means accepted, not queryable. Ingestion is
    asynchronous and took ~90s in testing, so a single immediate read is
    guaranteed to come back empty and would make a working write look broken.

    Two other things bite here, and both look identical from the caller --
    zero rows. The export window must reach back to where the spans were
    *dated*, not to now; and until ingestion has actually created the project,
    the export raises `unauthorized ... model does not exist` rather than
    returning nothing, which reads as a credentials problem and isn't one.
    """
    import logging
    import time

    # The exporter logs a full traceback at ERROR for the not-yet-created
    # project, which here is the expected state rather than a fault. Eight of
    # those would bury the one line that matters.
    for name in ("arize._exporter.client", "arize._flight.client"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    start = datetime.now(timezone.utc) - timedelta(days=days_back + 1)
    for attempt in range(POLL_ATTEMPTS):
        try:
            found = client.spans.export_to_df(
                space_id=settings.arize_space_id,
                project_name=target,
                start_time=start,
                end_time=datetime.now(timezone.utc) + timedelta(minutes=5),
            )
            if not found.empty:
                return found
            console.print(f"  [dim]nothing yet ({attempt + 1}/{POLL_ATTEMPTS})…[/dim]")
        except Exception as exc:  # noqa: BLE001 - the write already succeeded
            console.print(
                f"  [dim]project not queryable yet ({attempt + 1}/{POLL_ATTEMPTS}): "
                f"{type(exc).__name__}[/dim]"
            )
        if attempt < POLL_ATTEMPTS - 1:
            time.sleep(POLL_SECONDS)
    return None


@app.command()
def main(
    project: str = typer.Option("", help="Target project (default: <project>-backfill)"),
    days_back: int = typer.Option(30, help="How far in the past to place the history"),
    verify: bool = typer.Option(True, help="Read the spans back after logging"),
) -> None:
    settings = header(
        "02b",
        "Instrument: log spans directly, without OpenTelemetry",
        "spans client resource · view-and-manage-traces",
    )

    target = project or f"{settings.arize_project_name}-backfill"
    spans_df, evals_df = build_history(HISTORIC_TICKETS, days_back)

    console.print(
        f"Logging [bold]{len(spans_df)}[/bold] spans across "
        f"{spans_df['context.trace_id'].nunique()} traces into [bold]{target}[/bold], "
        f"dated {days_back} days ago.\n"
    )

    client = arize_client(settings)
    response = client.spans.log(
        space_id=settings.arize_space_id,
        project_name=target,
        dataframe=spans_df,
        evals_dataframe=evals_df,
    )
    console.print(f"[green]Accepted[/green] (HTTP {getattr(response, 'status_code', '?')}).")

    # Joined on span id rather than zipped positionally. The two frames happen
    # to be built in the same order today, which is exactly the kind of implicit
    # invariant that later mislabels every row in silence.
    summary = (
        spans_df[spans_df[SPAN_KIND] == "AGENT"]
        .merge(evals_df, on="context.span_id")
    )
    table(
        "Backfilled",
        ["trace", "question", "groundedness"],
        [
            [
                row["context.trace_id"][:12],
                row["attributes.input.value"][:48],
                row["eval.groundedness.label"],
            ]
            for _, row in summary.iterrows()
        ],
    )

    if verify:
        console.print("\nReading them back…")
        found = wait_for_spans(client, settings, target, days_back)
        if found is None:
            console.print(
                f"[yellow]Not queryable within {POLL_ATTEMPTS * POLL_SECONDS}s.[/yellow] "
                "That is slow but not necessarily wrong — check the project in the UI."
            )
        else:
            console.print(f"[green]{len(found)} spans[/green] queryable in {target}.")
            kinds = found[SPAN_KIND].value_counts() if SPAN_KIND in found else {}
            # Match the shape of a span id rather than testing for non-empty:
            # a root's `parent_id` comes back as NaN, which stringifies to
            # "nan" and would count as a parent under a length or truthiness
            # test -- reporting every root span as nested.
            parent_col = found["parent_id"] if "parent_id" in found else pd.Series(dtype=str)
            parents = int(
                parent_col.fillna("").astype(str).str.fullmatch(r"[0-9a-f]{16}").sum()
            )
            graded = int(found.get("eval.groundedness.score", pd.Series(dtype=float)).notna().sum())
            table(
                "Read back",
                ["property", "value"],
                [
                    ["span kinds", ", ".join(f"{k}×{v}" for k, v in dict(kinds).items()) or "-"],
                    ["spans with a parent", parents],
                    ["spans carrying the joined eval", graded],
                ],
            )

    look_at(
        f"Projects → {target}. It exists only because these spans created it.",
        "Open a trace: the AGENT → LLM nesting came from `parent_id`, not a tracer.",
        "The groundedness column is there too — evals_dataframe joined on span id "
        "during the same call.",
        f"Note the timestamps: {days_back} days back. Backfilled history sorts into "
        "place rather than piling up at 'now'.",
    )
    done(
        "poc/03_query_spans.py — analyse the live project",
        "[dim]This backfill project is not part of the rest of the tour.[/dim]",
    )


if __name__ == "__main__":
    app()

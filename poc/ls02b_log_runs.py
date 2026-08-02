#!/usr/bin/env python
"""Step ls02b -- Instrument: get runs into LangSmith without OpenTelemetry.

The LangSmith mirror of poc/02b, same premise: last quarter's conversations
are warehouse rows, not running processes. `batch_ingest_runs` is the other
door -- hand it run dicts and it ingests them directly, no tracer, no agent.

The fixture is 02b's own HISTORIC_TICKETS, ids derived deterministically from
the ticket (uuid5) so a re-run replaces the same runs instead of doubling the
history. Verdicts ride along as feedback, which is the LangSmith counterpart
of 02b logging an evals dataframe next to the spans.

The synthetic history goes to its own project (`<project>-backfill`), for
02b's reason verbatim: mixing backfilled rows into the project ls03 analyses
would quietly change every count that follows.

Docs: https://docs.smith.langchain.com/observability/how_to_guides/trace_without_langchain
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from importlib import import_module

import typer

from _common import console, done, header, table
from _ls_common import look_at_ls, ls_client, ls_project_id, require_langsmith, upsert_feedback

backfill = import_module("02b_log_spans")

app = typer.Typer(add_completion=False)


def run_uuid(*parts: str) -> str:
    """Deterministic run id -- LangSmith wants UUID format, not bare hex."""
    return str(uuid.uuid5(backfill._NAMESPACE, "ls|" + "|".join(parts)))


def dotted(start: datetime, run_id: str, parent: str = "") -> str:
    """LangSmith's ordering key: `<start><id>` segments joined by dots."""
    segment = f"{start.strftime('%Y%m%dT%H%M%S%f')}Z{run_id}"
    return f"{parent}.{segment}" if parent else segment


def build_history(tickets, days_back: int, project: str):
    """Run dicts (roots + LLM children) and the feedback rows to attach."""
    runs, feedback = [], []
    base = datetime.now(timezone.utc) - timedelta(days=days_back)

    for i, (question, answer, score, label) in enumerate(tickets):
        trace = run_uuid("trace", str(i))
        child = run_uuid("llm", str(i))
        start = base + timedelta(minutes=7 * i)
        llm_start = start + timedelta(milliseconds=120)
        llm_end = llm_start + timedelta(milliseconds=1500 + 90 * i)
        end = llm_end + timedelta(milliseconds=40)
        root_order = dotted(start, trace)

        runs.append(
            {
                "id": trace,
                "trace_id": trace,
                "dotted_order": root_order,
                "name": "legacy.ticket",
                "run_type": "chain",
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "inputs": {"input": question},
                "outputs": {"text": answer},
                "session_name": project,
                "extra": {
                    "metadata": {
                        "source": "warehouse-backfill",
                        "handled_by": "legacy-macro-system",
                        "session_id": f"legacy-{i:03d}",
                        "user_id": f"customer-{i:03d}",
                    }
                },
            }
        )
        runs.append(
            {
                "id": child,
                "trace_id": trace,
                "parent_run_id": trace,
                "dotted_order": dotted(llm_start, child, root_order),
                "name": "legacy.answer",
                "run_type": "llm",
                "start_time": llm_start.isoformat(),
                "end_time": llm_end.isoformat(),
                "inputs": {"input": question},
                "outputs": {"text": answer},
                "session_name": project,
                "extra": {
                    "metadata": {
                        "ls_model_name": "legacy-macro-v1",
                        "usage_metadata": {
                            "input_tokens": 180 + 12 * i,
                            "output_tokens": len(answer.split()),
                            "total_tokens": 180 + 12 * i + len(answer.split()),
                        },
                    }
                },
            }
        )
        feedback.append((trace, label, score))
    return runs, feedback


@app.command()
def main(
    project: str = typer.Option("", help="Target project (default: <project>-backfill)"),
    days_back: int = typer.Option(30, help="How far in the past to place the history"),
    verify: bool = typer.Option(True, help="Read the runs back after logging"),
) -> None:
    settings = header(
        "ls02b",
        "Instrument: log runs directly, without OpenTelemetry",
        "batch ingest · trace_without_langchain",
    )
    require_langsmith(settings, "direct run ingestion")

    target = project or f"{settings.langsmith_project}-backfill"
    client = ls_client(settings)

    runs, feedback = build_history(backfill.HISTORIC_TICKETS, days_back, target)
    console.print(
        f"Ingesting [bold]{len(runs)}[/bold] runs "
        f"({len(backfill.HISTORIC_TICKETS)} tickets × root + LLM child) into "
        f"[bold]{target}[/bold]…"
    )
    client.batch_ingest_runs(create=runs)

    # Feedback wants the project id, and ingestion *creates* the project
    # asynchronously -- 02b's lesson (accepted is not queryable) applies to the
    # project record too, so poll for it rather than failing on the first try.
    import time

    pid = ""
    for attempt in range(8):
        try:
            pid = ls_project_id(client, target)
            break
        except Exception:  # noqa: BLE001 - not ingested yet
            console.print(f"  [dim]project not visible yet ({attempt + 1}/8)…[/dim]")
            time.sleep(15)
    if not pid:
        console.print(
            "[red]The ingest was accepted but the project never appeared.[/red] "
            "No feedback can be attached; check the UI before trusting this backfill."
        )
        raise SystemExit(1)

    console.print("Attaching the legacy verdicts as feedback…")
    for run_id, label, score in feedback:
        upsert_feedback(
            client,
            run_id=run_id,
            key="groundedness",
            project_id=pid,
            score=float(score),
            value=label,
            comment="Backfilled verdict from the legacy quality review.",
        )
    console.print(f"[green]{len(feedback)} verdicts attached.[/green]")

    if verify:
        import time

        console.print("\nReading them back (ingestion is asynchronous)…")
        found = []
        for attempt in range(8):
            found = list(client.list_runs(project_name=target))
            if found:
                break
            console.print(f"  [dim]nothing yet ({attempt + 1}/8)…[/dim]")
            time.sleep(15)
        if not found:
            console.print(
                "[red]The runs were accepted but never became queryable.[/red] "
                "Accepted is not ingested — check the project in the UI before "
                "trusting this backfill."
            )
            raise SystemExit(1)
        roots = [r for r in found if r.parent_run_id is None]
        table(
            "Read back",
            ["what", "count"],
            [["runs", len(found)], ["tickets (roots)", len(roots)]],
        )
        console.print(
            "[green]The backfill is queryable[/green] — parent/child structure and "
            "all, no tracer involved."
        )

    look_at_ls(
        f"Tracing Projects → {target}: five conversations dated ~{days_back} days ago.",
        "Open one — the trace tree (chain → llm) was rebuilt from dotted_order, "
        "proving structure survives direct ingestion.",
        "The groundedness feedback came in with the backfill: history arrives "
        "already graded, no second pass needed.",
    )
    done("The main tour continues in poc/ls03_query_runs.py")


if __name__ == "__main__":
    app()

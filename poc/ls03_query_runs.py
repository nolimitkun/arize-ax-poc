#!/usr/bin/env python
"""Step ls03 -- Observe: query LangSmith runs and find the failing traces.

The LangSmith mirror of poc/03. Same failure definitions -- they are imported
from 03, not copied -- but the trajectory comes out of the run *tree* instead
of span attributes: LangSmith's OTel ingest drops custom `copilot.*`
attributes, so tools-called and docs-retrieved are rebuilt from the tool and
retriever child runs (see _ls_common.turn_runs_to_df).

What it finds is written back as run tags, so the Runs table can be filtered
by failure mode -- the same "don't leave the analysis in a dataframe" rule as
step 03, through LangSmith's mechanism for it.

The output feeds ls04 (feedback), ls06 (annotation queue) and ls07 (dataset).

Docs: https://docs.smith.langchain.com/observability/how_to_guides/export_traces
"""

from __future__ import annotations

from importlib import import_module

import typer

from _common import console, done, header, save, table, window
from _ls_common import look_at_ls, ls_client, require_langsmith, turn_runs_to_df

# The definitions of "a failure" belong to step 03; importing them is the
# point -- both platforms grade the same traffic by the same rules.
observe = import_module("03_query_spans")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    hours: int = typer.Option(24, help="How far back to query"),
    limit: int = typer.Option(0, help="Cap turns for a quick look (0 = no cap)"),
    skip_tags: bool = typer.Option(
        False, help="Don't write the failure classification back as run tags"
    ),
) -> None:
    settings = header(
        "ls03",
        "Observe: query LangSmith runs, find the failing traces",
        "export_traces · filter_traces_in_application",
    )
    require_langsmith(settings, "run queries")

    client = ls_client(settings)
    start, _ = window(hours)

    console.print(f"Querying runs from the last {hours}h…")
    runs = list(
        client.list_runs(project_name=settings.langsmith_project, start_time=start)
    )
    if not runs:
        console.print(
            "[red]No runs returned.[/red] Run poc/01_trace.py first with LangSmith "
            "enabled, and give ingestion a minute."
        )
        raise SystemExit(1)

    kinds: dict[str, int] = {}
    for r in runs:
        kinds[r.run_type] = kinds.get(r.run_type, 0) + 1
    console.print(f"[green]{len(runs)} runs.[/green]\n")
    table("Runs by type", ["run_type", "count"], sorted(kinds.items(), key=lambda x: -x[1]))

    roots = [r for r in runs if r.name == "copilot.turn"]
    if not roots:
        console.print("[red]No `copilot.turn` runs found.[/red] Was poc/01 run?")
        raise SystemExit(1)
    children: dict[str, list] = {}
    for r in runs:
        if r.parent_run_id is not None:
            children.setdefault(str(r.trace_id), []).append(r)

    turns = turn_runs_to_df(roots, children)
    turns = turns.sort_values("start_time", kind="stable").reset_index(drop=True)

    # Experiment targets are double-traced into this project (the ls08 caveat):
    # every arm's answer_only call emits its own copilot.turn tree. Counting
    # those as production traffic would feed ls07's dataset the experiment's
    # own answers -- the analysis grading its own homework. The engine invents
    # a session_id when none is given, so that can't be the filter; what only
    # real traffic has is a caller-supplied user (01's personas, 02's demo
    # user). Experiment and probe calls run as "anonymous".
    experiment_turns = int((turns["user_id"].isin(["", "anonymous"])).sum())
    if experiment_turns:
        console.print(
            f"[dim]Excluding {experiment_turns} anonymous turn(s) — experiment/"
            "probe traffic double-traced into this project, not production "
            "conversations.[/dim]\n"
        )
        turns = turns[~turns["user_id"].isin(["", "anonymous"])].reset_index(drop=True)
    if limit:
        turns = turns.head(limit)

    # Expected behaviour lives in the local fixture, joined back by question
    # text -- identical to step 03.
    from copilot.agent import load_questions

    expectations = {q["question"]: q for q in load_questions()}

    failures_col, ids, expected_all, behaviors = [], [], [], []
    for _, row in turns.iterrows():
        meta = expectations.get(row["question"], {})
        expected_tools = meta.get("expected_tools", [])
        tools = [t for t in row["tool_calls"].split(",") if t]
        docs = [d for d in row["retrieved_doc_ids"].split(",") if d]
        failures = []
        if observe.check_ungrounded(row["question"], row["answer"], docs):
            failures.append("hallucination")
        if observe.check_wrong_tool(expected_tools, tools):
            failures.append("wrong_tool")
        if observe.check_missing_escalation(expected_tools, tools):
            failures.append("missing_escalation")
        if observe.check_verbose(row["answer"]):
            failures.append("verbosity")
        failures_col.append(",".join(failures))
        ids.append(meta.get("id", ""))
        expected_all.append(",".join(expected_tools))
        behaviors.append(meta.get("expected_behavior", ""))

    turns["question_id"] = ids
    turns["expected_tools"] = expected_all
    turns["expected_behavior"] = behaviors
    turns["answer_words"] = turns["answer"].apply(lambda a: len(str(a).split()))
    turns["failures"] = failures_col
    turns["is_failure"] = [bool(f) for f in failures_col]

    counts: dict[str, int] = {}
    for entry in turns["failures"]:
        for f in filter(None, entry.split(",")):
            counts[f] = counts.get(f, 0) + 1
    total, failing = len(turns), int(turns["is_failure"].sum())
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

    save("ls03_turns.parquet", turns.drop(columns=["start_time"]).assign(
        start_time=turns["start_time"].astype(str)
    ))
    save("ls03_failures.parquet", turns[turns["is_failure"]].assign(
        start_time=turns[turns["is_failure"]]["start_time"].astype(str)
    ))

    if not skip_tags:
        console.print("\nTagging the runs with what was found…")
        tagged, already, errors = 0, 0, []
        for _, r in turns.iterrows():
            modes = [m for m in r["failures"].split(",") if m]
            # "failure:none" rather than no tag -- an untagged run is
            # indistinguishable from an unexamined one, and "no failures"
            # is a finding (the same reasoning as step 03's "none").
            tags = [f"failure:{m}" for m in modes] or ["failure:none"]
            try:
                client.update_run(r["span_id"], tags=tags)
                tagged += 1
            except Exception as exc:  # noqa: BLE001 - handled per run below
                # The ingest API takes exactly one update per run: a run tagged
                # on a previous pass answers 409 forever after. That is "still
                # tagged", not a failure -- but it must not abort the loop, or
                # one old run would silently cost every new run its tags.
                if "409" in str(exc) or "Conflict" in type(exc).__name__:
                    already += 1
                else:
                    errors.append(f"{type(exc).__name__}: {exc}")
        console.print(
            f"[green]Tagged {tagged} runs[/green] with `failure:*` — filterable "
            "in the Runs table."
            + (f" [dim]({already} kept tags from an earlier pass.)[/dim]" if already else "")
        )
        if errors:
            console.print(
                f"[yellow]{len(errors)} run(s) could not be tagged.[/yellow] "
                f"First error: {errors[0][:200]}"
            )

    look_at_ls(
        "Tracing Projects → the project → Runs: filter `Name = copilot.turn`, sort "
        "by latency, open the slowest turn.",
        "Filter tag `failure:hallucination` — the classification this step computed "
        "is now a run tag, one filter away instead of one parquet file away.",
        "Open a trace: the same tree Arize shows, in LangSmith's rendering — "
        "chain → llm/retriever/tool children.",
    )
    done(
        "poc/ls04_offline_evals.py — LLM-as-judge over these runs, as feedback",
        "poc/ls07_dataset.py — turn the failures into a LangSmith dataset",
    )


if __name__ == "__main__":
    app()

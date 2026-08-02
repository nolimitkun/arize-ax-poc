#!/usr/bin/env python
"""Step ls06 -- Evaluate: an annotation queue, and whether the judge agrees.

The LangSmith mirror of poc/06, and the closest 1:1 in the whole port --
annotation queues are a native LangSmith feature. Same shape:

  1. create a queue (with the rubric as queue instructions) seeded with the
     failing runs
  2. write simulated human labels as `human_groundedness` feedback
  3. report judge-vs-human agreement

The simulated labels are poc/06's own (imported): same fixture-derived ground
truth, same deterministic ~8% reviewer noise, so ls06b's train/holdout split
measures against stable labels on either platform.

Docs: https://docs.smith.langchain.com/evaluation/how_to_guides/annotation_queues
"""

from __future__ import annotations

from importlib import import_module

import typer

from _common import console, done, header, load, save, table
from _ls_common import look_at_ls, ls_client, ls_project_id, require_langsmith, upsert_feedback

anno = import_module("06_annotations")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    sample: int = typer.Option(20, help="How many runs to put in the review queue"),
    skip_queue: bool = typer.Option(False, help="Skip queue creation, just write labels"),
    label_all: bool = typer.Option(
        True, help="Simulate labels for every turn, not just the queued ones"
    ),
    prune_duplicates: bool = typer.Option(
        False,
        help="Also delete labels left by runs that predate deterministic "
        "feedback ids (one extra API call per run)",
    ),
) -> None:
    settings = header(
        "ls06",
        "Evaluate: annotation queue, labels, and judge agreement",
        "annotation_queues · attach_user_feedback",
    )
    require_langsmith(settings, "annotation queues")

    client = ls_client(settings)
    pid = ls_project_id(client, settings.langsmith_project)
    turns = load("ls03_turns.parquet")

    ranked = turns.sort_values("is_failure", ascending=False).head(sample).copy()
    labelled = turns.copy() if label_all else ranked
    console.print(
        f"Selected [bold]{len(ranked)}[/bold] runs for the review queue "
        f"({int(ranked['is_failure'].sum())} already flagged by code checks)."
    )
    if label_all and len(labelled) > len(ranked):
        console.print(
            f"Simulating labels for all [bold]{len(labelled)}[/bold] turns "
            f"[dim](--no-label-all to label only the queued {len(ranked)})[/dim]"
        )
    console.print()

    # ---- 1. Annotation queue --------------------------------------------
    # Project-scoped name, same reasoning as 06's queue_name (imported): queues
    # are workspace-level in LangSmith too, and a fixed name would leave the
    # second project's tour "reusing" a queue full of the first project's runs.
    name = anno.queue_name(settings.langsmith_project)
    if not skip_queue:
        try:
            existing = next(iter(client.list_annotation_queues(name=name)), None)
            if existing is not None:
                queue_id = existing.id
                console.print(f"[dim]Queue {name!r} already exists; adding runs to it.[/dim]")
            else:
                queue = client.create_annotation_queue(
                    name=name,
                    description=f"Groundedness review of {settings.langsmith_project} turns",
                    rubric_instructions=(
                        "Read the question and the assistant's answer. Mark 'hallucinated' "
                        "if the answer states any Nimbus policy, number or timeframe that "
                        "isn't in the retrieved documentation -- even if it sounds right. "
                        "Declining to answer an undocumented question is 'grounded'."
                    ),
                )
                queue_id = queue.id
                console.print(f"[green]Created review queue[/green] {name} ({queue_id})")
            client.add_runs_to_annotation_queue(
                queue_id,
                runs=[
                    {"run_id": row["span_id"], "session_id": pid, "start_time": row["start_time"]}
                    for _, row in ranked.iterrows()
                ],
            )
            # List back rather than trust the add call -- the lesson every
            # queue in this repo has taught: creation succeeding says nothing
            # about the queue containing what a reviewer will actually see.
            queued = list(client.list_runs_from_annotation_queue(queue_id))
            console.print(
                f"[green]{len(queued)} runs in the queue[/green] (verified by listing back)."
            )
            if not queued:
                console.print(
                    "[yellow]The queue exists but lists back empty — a reviewer "
                    "opening it sees nothing. Check the run ids.[/yellow]"
                )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Queue not created: {type(exc).__name__}: {exc}[/yellow]")
            console.print(
                "  [dim]The labels below still land on the runs, so the agreement "
                "number stands — but there is nothing for a reviewer to open.[/dim]"
            )

    # ---- 2. Write human labels (simulated, poc/06's stand-in) -----------
    console.print("\n[bold]Writing (simulated) human labels[/bold]")
    labels = labelled.apply(anno.simulated_human_label, axis=1, result_type="expand")
    import pandas as pd

    annotations = pd.DataFrame(
        {
            "context.span_id": labelled["span_id"].astype(str).values,
            f"annotation.{anno.ANNOTATION_NAME}.label": labels[0].values,
            f"annotation.{anno.ANNOTATION_NAME}.score": labels[1].astype(float).values,
            f"annotation.{anno.ANNOTATION_NAME}.text": labels[2].values,
        }
    )
    for _, row in annotations.iterrows():
        upsert_feedback(
            client,
            run_id=row["context.span_id"],
            key=anno.ANNOTATION_NAME,
            project_id=pid,
            score=float(row[f"annotation.{anno.ANNOTATION_NAME}.score"]),
            value=str(row[f"annotation.{anno.ANNOTATION_NAME}.label"]),
            comment=str(row[f"annotation.{anno.ANNOTATION_NAME}.text"]),
            prune=prune_duplicates,
        )
    console.print(f"[green]Logged {len(annotations)} human labels as feedback.[/green]")
    save("ls06_annotations.parquet", annotations)

    # ---- 3. Judge vs human ----------------------------------------------
    try:
        judge = load("ls04_evals.parquet")
    except SystemExit:
        console.print("[yellow]No ls04_evals.parquet — skipping agreement.[/yellow]")
        done()
        return

    judge_col = "eval.groundedness.label"
    if judge_col not in judge.columns:
        console.print(f"[yellow]{judge_col} not present (was ls04 run with --skip-llm?).[/yellow]")
        done()
        return

    merged = annotations.merge(
        judge[["context.span_id", judge_col]], on="context.span_id", how="inner"
    )
    human_col = f"annotation.{anno.ANNOTATION_NAME}.label"
    agree = (merged[human_col] == merged[judge_col]).sum()
    n = len(merged)
    both_bad = ((merged[human_col] == "hallucinated") & (merged[judge_col] == "hallucinated")).sum()
    judge_only = ((merged[human_col] == "grounded") & (merged[judge_col] == "hallucinated")).sum()
    human_only = ((merged[human_col] == "hallucinated") & (merged[judge_col] == "grounded")).sum()

    table(
        "Judge vs human (label = hallucinated)",
        ["metric", "value"],
        [
            ["compared runs", n],
            ["agreement", f"{100 * agree / n:.0f}%" if n else "n/a"],
            ["both flagged", both_bad],
            ["judge flagged, human didn't (false positive)", judge_only],
            ["human flagged, judge didn't (false negative)", human_only],
        ],
    )
    if n and agree / n < 0.8:
        console.print(
            "\n[yellow]Agreement below 80%.[/yellow] The fix is to tighten the judge "
            "template using the disagreeing examples — not to trust the judge's "
            "aggregate score yet. That is ls06b.\n"
        )

    look_at_ls(
        "Annotation Queues → the queue, with the rubric as its instructions.",
        "Open a queued run and label it yourself; the feedback lands on the same run.",
        f"A run carrying both `groundedness` and `{anno.ANNOTATION_NAME}` feedback — "
        "side by side is how you decide whether the judge is trustworthy.",
    )
    done("poc/ls06b_align_judge.py — align the judge to these labels")


if __name__ == "__main__":
    app()

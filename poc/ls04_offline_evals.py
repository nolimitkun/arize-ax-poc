#!/usr/bin/env python
"""Step ls04 -- Evaluate: judges + code evaluators, logged back as feedback.

The LangSmith mirror of poc/04. The evaluators are poc/04's own -- imported,
not copied -- so both platforms grade with one definition of groundedness.
What changes is the write-back: Arize takes a dataframe of `eval.*` columns
per span; LangSmith takes one `create_feedback` per run per key, and the
scores surface in the Runs table and feedback charts.

Docs: https://docs.smith.langchain.com/evaluation/how_to_guides/attach_user_feedback
"""

from __future__ import annotations

from importlib import import_module

import pandas as pd
import typer

from _common import console, done, header, load, save, table
from _ls_common import look_at_ls, ls_client, ls_project_id, require_langsmith, upsert_feedback

# One definition of every evaluator, owned by step 04.
offline = import_module("04_offline_evals")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    judge_model: str = typer.Option("deepseek-v4-pro", help="Model backing the LLM judge"),
    skip_llm: bool = typer.Option(False, help="Run only the code evaluators"),
    limit: int = typer.Option(0, help="Evaluate only the first N turns"),
    prune_duplicates: bool = typer.Option(
        False,
        help="Also delete verdicts left by runs that predate deterministic "
        "feedback ids (one extra API call per run)",
    ),
) -> None:
    settings = header(
        "ls04",
        "Evaluate: LLM-as-judge + code evaluators, logged back as feedback",
        "attach_user_feedback · evaluation concepts",
    )
    require_langsmith(settings, "run feedback")

    turns = load("ls03_turns.parquet")
    if limit:
        turns = turns.head(limit)
    console.print(f"Evaluating [bold]{len(turns)}[/bold] agent turns.\n")

    results = pd.DataFrame({"context.span_id": turns["span_id"].astype(str)})
    judge_results = pd.DataFrame()

    # ---- code evaluators (poc/04's, verbatim) ---------------------------
    console.print("[bold]Code evaluators[/bold] (deterministic)")
    for name, fn in offline.CODE_EVALUATORS.items():
        applied = turns.apply(fn, axis=1, result_type="expand")
        results[f"eval.{name}.label"] = applied[0].values
        results[f"eval.{name}.score"] = applied[1].astype(float).values
        results[f"eval.{name}.explanation"] = applied[2].values
        console.print(f"  {name:<24} mean score {applied[1].astype(float).mean():.2f}")

    # ---- LLM judge (poc/04's evaluator stack) ---------------------------
    if not skip_llm:
        console.print(f"\n[bold]LLM-as-a-judge[/bold] (Phoenix Evals, {judge_model})")
        from phoenix.evals import evaluate_dataframe

        from copilot.kb import context_for_ids

        judge_input = turns[["question", "answer"]].copy()
        judge_input["retrieved_context"] = turns["retrieved_doc_ids"].apply(
            lambda ids: context_for_ids([i for i in str(ids).split(",") if i])
        )
        graded = evaluate_dataframe(
            dataframe=judge_input,
            evaluators=offline.build_llm_evaluators(judge_model, settings),
        )
        for note in offline.judge_failures(graded, offline.GROUNDEDNESS):
            console.print(f"  [yellow]{note}[/yellow]")

        parsed = offline.parse_judge_output(graded, offline.GROUNDEDNESS)
        if parsed is None:
            console.print(f"[red]Judge produced no '{offline.GROUNDEDNESS}_score' column[/red]")
        else:
            labels, scores, expl = parsed
            ok = labels.notna() & scores.notna()
            judge_results = pd.DataFrame(
                {
                    "context.span_id": results["context.span_id"][ok.values].values,
                    f"eval.{offline.GROUNDEDNESS}.label": labels[ok].values,
                    f"eval.{offline.GROUNDEDNESS}.score": scores[ok].values,
                    f"eval.{offline.GROUNDEDNESS}.explanation": expl[ok].fillna("").values,
                }
            )
            skipped = int((~ok).sum())
            if skipped:
                console.print(
                    f"  [yellow]{skipped} row(s) had no usable judge verdict and get "
                    "no feedback.[/yellow]"
                )
            if len(judge_results):
                console.print(
                    f"  {offline.GROUNDEDNESS:<24} mean score "
                    f"{judge_results[f'eval.{offline.GROUNDEDNESS}.score'].mean():.2f} "
                    f"({len(judge_results)} rows)"
                )

    # ---- log back to LangSmith ------------------------------------------
    client = ls_client(settings)
    pid = ls_project_id(client, settings.langsmith_project)
    console.print("\nWriting feedback onto the runs…")

    def upload(frame: pd.DataFrame, what: str) -> None:
        if not len(frame):
            return
        names = sorted(
            {c.split(".")[1] for c in frame.columns if c.startswith("eval.")}
        )
        written = 0
        for _, row in frame.iterrows():
            for name in names:
                score = row.get(f"eval.{name}.score")
                if pd.isna(score):
                    continue
                upsert_feedback(
                    client,
                    run_id=row["context.span_id"],
                    key=name,
                    project_id=pid,
                    score=float(score),
                    value=str(row.get(f"eval.{name}.label", "")),
                    comment=str(row.get(f"eval.{name}.explanation", "")),
                    prune=prune_duplicates,
                )
                written += 1
        console.print(f"[green]Logged {written} feedback entries[/green] ({what}).")

    upload(results, "code evaluators")
    upload(judge_results, offline.GROUNDEDNESS)

    combined = (
        results.merge(judge_results, on="context.span_id", how="left")
        if len(judge_results)
        else results
    )
    save("ls04_evals.parquet", combined)

    score_cols = [c for c in combined.columns if c.endswith(".score")]
    table(
        "Evaluation summary",
        ["evaluator", "mean score", "graded", "failing turns"],
        [
            [
                c.removeprefix("eval.").removesuffix(".score"),
                f"{combined[c].mean():.2f}",
                int(combined[c].notna().sum()),
                int((combined[c] < 1.0).sum()),
            ]
            for c in score_cols
        ],
    )

    look_at_ls(
        "Runs table → the feedback keys are now columns; sort by `groundedness`.",
        "Open a failing run → Feedback tab → read the judge's explanation — that "
        "text is what tells you how to rewrite the prompt.",
        "Project home → the feedback charts now have data (mean score over time).",
    )
    done(
        "poc/ls04b_thread_evals.py — grade whole conversations",
        "poc/ls06_annotations.py — human labels via an annotation queue",
    )


if __name__ == "__main__":
    app()

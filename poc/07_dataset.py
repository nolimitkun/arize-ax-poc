#!/usr/bin/env python
"""Step 07 -- Improve: turn the failing traces into a versioned dataset.

Spot-checking a prompt change against three examples you remember is how you
convince yourself of an improvement that isn't real. A dataset makes the change
measurable: same inputs, same evaluators, before and after.

Failures come from the *evaluation results* of step 04, not from step 03's
heuristics. That ordering matters and it is what the AX improve flow prescribes:
the code checks in step 03 are keyword-narrow (`check_ungrounded` only fires on
refund phrasing) and found 1 hallucination across 39 turns, while the LLM judge
grading every answer flagged ~21. Building a dataset from the heuristics alone
would throw away most of the real signal.

A control group of turns that already passed is included too -- without it you
can "fix" hallucination by making the agent refuse everything and never notice.

Docs: https://arize.com/docs/ax/improve/build-a-dataset
"""

from __future__ import annotations

import typer

from _common import arize_client, console, done, header, load, look_at, save, table

app = typer.Typer(add_completion=False)

DATASET_NAME = "copilot-failures"

# All eval columns, mapped to the failure they represent. Score < 1.0 is a
# failure for each (they are all maximize-direction).
FAILING_EVALS = {
    "eval.groundedness.score": "hallucination",
    "eval.conciseness.score": "verbosity",
    "eval.tool_selection.score": "wrong_tool",
    "eval.escalation_appropriate.score": "missing_escalation",
}

# ...but only these two decide whether a turn is *selected* into the dataset,
# because they are what step 08 re-measures. Selecting on tool_selection too
# would sweep in 25 of 39 turns that step 08 has no evaluator for -- it would
# leave no control group at all, and the experiment could not show movement on
# them either way. The other verdicts are still recorded on each example.
SELECTION_EVALS = ("eval.groundedness.score", "eval.conciseness.score")


def merge_eval_verdicts(turns):
    """Attach step 04's verdicts and derive `selected_failure` per turn.

    Falls back to step 03's `is_failure` heuristics when no eval results are
    present, so the script still does something sensible on a fresh checkout --
    but the eval path is the one that finds the real failures.
    """
    try:
        evals = load("04_evals.parquet")
    except SystemExit:
        console.print(
            "[yellow]No 04_evals.parquet — falling back to step 03's heuristics.[/yellow]\n"
            "[dim]Run poc/04_offline_evals.py for a dataset built from eval verdicts.[/dim]"
        )
        turns = turns.copy()
        turns["selected_failure"] = turns.apply(
            lambda r: r["failures"] if r["is_failure"] else "", axis=1
        )
        turns["all_failures"] = turns["selected_failure"]
        return turns

    score_cols = [c for c in FAILING_EVALS if c in evals.columns]
    merged = turns.merge(
        evals[["context.span_id", *score_cols]],
        left_on="span_id",
        right_on="context.span_id",
        how="left",
    )

    def modes(row, columns) -> str:
        # NaN means "the judge never graded this turn", which is not evidence of
        # a failure -- treat it as unknown rather than sweeping it in.
        return ",".join(
            FAILING_EVALS[col]
            for col in columns
            if col in row and row[col] == row[col] and row[col] < 1.0
        )

    selecting = [c for c in SELECTION_EVALS if c in score_cols]
    merged["selected_failure"] = merged.apply(lambda r: modes(r, selecting), axis=1)
    merged["all_failures"] = merged.apply(lambda r: modes(r, score_cols), axis=1)

    graded = merged[score_cols].notna().any(axis=1).sum() if score_cols else 0
    console.print(
        f"[dim]{graded}/{len(merged)} turns graded. Selecting on "
        f"{', '.join(c.split('.')[1] for c in selecting)}; also recording "
        f"{', '.join(c.split('.')[1] for c in score_cols if c not in selecting)}.[/dim]"
    )
    return merged


@app.command()
def main(
    name: str = typer.Option(DATASET_NAME, help="Dataset name in Arize"),
    controls: int = typer.Option(12, help="Passing turns to include as a control group"),
) -> None:
    settings = header(
        "07",
        "Improve: build a dataset from the failing traces",
        "build-a-dataset",
    )

    from copilot.agent import load_questions

    turns = load("03_turns.parquet")
    expectations = {q["id"]: q for q in load_questions()}

    turns = merge_eval_verdicts(turns)

    failures = turns[turns["selected_failure"] != ""].copy()
    passing = turns[turns["selected_failure"] == ""].head(controls).copy()

    if failures.empty:
        console.print(
            "[yellow]No failures found.[/yellow] Run poc/04_offline_evals.py first "
            "so the judge verdicts exist, and check poc/01 ran with prompt_version=v1."
        )
        raise SystemExit(1)

    console.print(
        f"{len(failures)} failing turns + {len(passing)} controls "
        f"= [bold]{len(failures) + len(passing)}[/bold] examples\n"
    )

    examples = []
    for _, row in [*failures.iterrows(), *passing.iterrows()]:
        meta = expectations.get(row["question_id"], {})
        examples.append(
            {
                # The task in step 08 reads `question`; the evaluators read the
                # `expected_*` fields. Keeping both on the example is what lets
                # one dataset drive several evaluators.
                "question": row["question"],
                "question_id": row["question_id"],
                "expected_behavior": row["expected_behavior"],
                "expected_tools": row["expected_tools"],
                "topic": meta.get("topic", ""),
                "failure_mode": row["selected_failure"],
                # Everything the evaluators flagged, including modes step 08
                # doesn't re-measure -- useful context when reading the dataset.
                "all_failures": row.get("all_failures", row["selected_failure"]),
                "is_control": row["selected_failure"] == "",
                # Baseline answer, so the dataset also records what v1 did.
                "baseline_answer": row["answer"],
                "baseline_tool_calls": row["tool_calls"],
                "source_span_id": row["span_id"],
                "source_trace_id": row["trace_id"],
            }
        )

    client = arize_client(settings)
    try:
        dataset = client.datasets.create(
            name=name,
            space=settings.arize_space_name,
            examples=examples,
        )
        console.print(
            f"[green]Created dataset[/green] {name} ({getattr(dataset, 'id', '?')}) "
            f"with {len(examples)} examples"
        )
    except Exception as exc:  # noqa: BLE001 - re-runs hit "already exists"
        console.print(f"[yellow]Create failed ({exc}); appending to the existing dataset.[/yellow]")
        client.datasets.append_examples(
            dataset=name,
            space=settings.arize_space_name,
            examples=examples,
        )
        dataset = client.datasets.get(dataset=name, space=settings.arize_space_name)
        console.print(f"[green]Appended {len(examples)} examples[/green] to {name}")

    import pandas as pd

    df = pd.DataFrame(examples)
    save("07_dataset.parquet", df)

    breakdown = (
        df[~df["is_control"]]["failure_mode"].str.split(",").explode().value_counts()
    )
    table(
        "Dataset composition",
        ["group", "examples"],
        [
            *[[f"failure: {mode}", count] for mode, count in breakdown.items() if mode],
            ["control (already passing)", int(df["is_control"].sum())],
            ["total", len(df)],
        ],
    )

    look_at(
        f"Datasets → {name}. Each example carries the question plus its expected behaviour.",
        "Dataset versions — appending creates a new version, so an experiment is "
        "always pinned to known inputs.",
        "[bold]Open the dataset in Prompt Playground[/bold] (UI only): edit the system "
        "prompt against these exact inputs and see answers change side by side. "
        "That is the manual version of what step 08 automates.",
    )
    done("poc/08_experiments.py — run v1 vs v2 against this dataset")


if __name__ == "__main__":
    app()

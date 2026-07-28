#!/usr/bin/env python
"""Step 07 -- Improve: turn the failing traces into a versioned dataset.

Spot-checking a prompt change against three examples you remember is how you
convince yourself of an improvement that isn't real. A dataset makes the change
measurable: same inputs, same evaluators, before and after.

The dataset is built from the failures found in step 03, plus a control group of
turns that already worked -- without the control you can "fix" hallucination by
making the agent refuse everything and never notice.

Docs: https://arize.com/docs/ax/improve/build-a-dataset
"""

from __future__ import annotations

import typer

from _common import arize_client, console, done, header, load, look_at, save, table

app = typer.Typer(add_completion=False)

DATASET_NAME = "copilot-failures"


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

    failures = turns[turns["is_failure"]].copy()
    passing = turns[~turns["is_failure"]].head(controls).copy()

    if failures.empty:
        console.print(
            "[yellow]No failures found in 03_turns.parquet.[/yellow] "
            "Was poc/01 run with prompt_version=v1?"
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
                "failure_mode": row["failures"],
                "is_control": not bool(row["is_failure"]),
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
        df[~df["is_control"]]["failure_mode"]
        .str.split(",")
        .explode()
        .value_counts()
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

#!/usr/bin/env python
"""Step ls07 -- Improve: turn the failing traces into a versioned dataset.

The LangSmith mirror of poc/07, selection logic included (imported from it):
failures come from ls04's *eval verdicts*, not ls03's keyword heuristics, and
a control group of passing turns rides along so a fix that breaks the passing
cases is visible.

LangSmith shape: `inputs` carry what the target consumes (the question),
`outputs` carry the reference the evaluators grade against (expected
behaviour/tools), and everything else -- baseline answer, failure modes,
source run -- is metadata. Human verdicts from ls06 are folded in as example
metadata; note the asymmetry with Arize, which has a separate dataset-
annotation channel (whose write-path 07 documents as unverifiable anyway).

Docs: https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_programmatically
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from importlib import import_module

import pandas as pd
import typer

from _common import console, done, header, load, save, table
from _ls_common import look_at_ls, ls_client, require_langsmith

ds = import_module("07_dataset")

app = typer.Typer(add_completion=False)

DATASET_NAME = "copilot-failures-ls"


def existing_examples(client, name: str) -> dict[str, str]:
    """`source_span_id` -> example id, for examples already in the dataset."""
    index: dict[str, str] = {}
    for example in client.list_examples(dataset_name=name):
        span = str((example.metadata or {}).get("source_span_id", ""))
        if span:
            index[span] = str(example.id)
    return index


@app.command()
def main(
    name: str = typer.Option(DATASET_NAME, help="Dataset name in LangSmith"),
    controls: int = typer.Option(12, help="Passing turns to include as a control group"),
) -> None:
    settings = header(
        "ls07",
        "Improve: build a dataset from the failing traces",
        "manage_datasets_programmatically",
    )
    require_langsmith(settings, "datasets")

    from copilot.agent import load_questions

    turns = ds.merge_eval_verdicts(load("ls03_turns.parquet"), evals_name="ls04_evals.parquet")
    expectations = {q["id"]: q for q in load_questions()}

    failures = turns[turns["selected_failure"] != ""].copy()
    passing = turns[turns["selected_failure"] == ""].head(controls).copy()
    if failures.empty:
        console.print(
            "[yellow]No failures found.[/yellow] Run poc/ls04_offline_evals.py first "
            "so the judge verdicts exist, and check poc/01 ran with prompt_version=v1."
        )
        raise SystemExit(1)

    batch = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    console.print(
        f"{len(failures)} failing turns + {len(passing)} controls "
        f"= [bold]{len(failures) + len(passing)}[/bold] examples "
        f"[dim](batch {batch})[/dim]\n"
    )

    examples = []
    for _, row in [*failures.iterrows(), *passing.iterrows()]:
        meta = expectations.get(row["question_id"], {})
        examples.append(
            {
                "inputs": {"question": row["question"]},
                # Reference outputs: what the evaluators in ls08 grade against.
                "outputs": {
                    "expected_behavior": row["expected_behavior"],
                    "expected_tools": row["expected_tools"],
                },
                "metadata": {
                    "question_id": row["question_id"],
                    "topic": meta.get("topic", ""),
                    "failure_mode": row["selected_failure"],
                    "all_failures": row.get("all_failures", row["selected_failure"]),
                    "is_control": row["selected_failure"] == "",
                    "baseline_answer": row["answer"],
                    "baseline_tool_calls": row["tool_calls"],
                    "source_span_id": row["span_id"],
                    "source_trace_id": row["trace_id"],
                    # Which traffic sample this example came from, so ls08 can
                    # measure one sample rather than every sample ever
                    # collected. Step 07 needs a whole dataset per batch --
                    # Arize's experiments.run takes a name and no row filter --
                    # while LangSmith's evaluate() accepts an example list, so
                    # here the same intent costs one metadata field.
                    "batch": batch,
                },
                "split": "control" if row["selected_failure"] == "" else "failure",
            }
        )

    client = ls_client(settings)
    if not client.has_dataset(dataset_name=name):
        dataset = client.create_dataset(
            name,
            description=(
                "Failing turns of the Nimbus copilot (selected on eval verdicts) "
                "plus a passing control group. Built by poc/ls07_dataset.py."
            ),
        )
        console.print(f"[green]Created dataset[/green] {name} ({dataset.id})")
        fresh = examples
        known: dict[str, str] = {}
    else:
        # Append only what is genuinely new -- the same dedup 07 had to learn:
        # appending the whole batch per run silently triples every experiment.
        known = existing_examples(client, name)
        fresh = [
            e for e in examples if str(e["metadata"]["source_span_id"]) not in known
        ]
        console.print(
            f"[dim]Dataset exists with {len(known)} known source runs; "
            f"{len(fresh)} of {len(examples)} examples are new.[/dim]"
        )

    if fresh:
        client.create_examples(dataset_name=name, examples=fresh)
        console.print(f"[green]Added {len(fresh)} examples[/green] to {name}.")
    else:
        console.print(f"[green]All {len(examples)} examples are already in {name}[/green].")

    # The dataset accumulates across runs, which is the point -- coverage
    # grows -- but it means an experiment over the whole thing reports a
    # number blended across traffic samples. Say how many samples are in
    # there, so the blend is visible rather than assumed away.
    batches = Counter(
        str((e.metadata or {}).get("batch", "(unstamped)"))
        for e in client.list_examples(dataset_name=name)
    )
    if len(batches) > 1:
        spread = ", ".join(f"{k}={v}" for k, v in sorted(batches.items()))
        console.print(
            f"\n[dim]{name} now holds {sum(batches.values())} examples from "
            f"{len(batches)} traffic samples: {spread}.\n"
            f"poc/ls08_experiments.py --batch {batch} measures this run alone; "
            "`--batch latest` finds it automatically, `--batch all` keeps the "
            "blended view.[/dim]"
        )

    flat = pd.DataFrame(
        {
            "question": [e["inputs"]["question"] for e in examples],
            **{
                k: [e["metadata"][k] for e in examples]
                for k in ("question_id", "failure_mode", "is_control", "source_span_id")
            },
            "expected_behavior": [e["outputs"]["expected_behavior"] for e in examples],
            "expected_tools": [e["outputs"]["expected_tools"] for e in examples],
        }
    )
    save("ls07_dataset.parquet", flat)

    breakdown = (
        flat[~flat["is_control"]]["failure_mode"].str.split(",").explode().value_counts()
    )
    table(
        "Dataset composition",
        ["group", "examples"],
        [
            *[[f"failure: {mode}", count] for mode, count in breakdown.items() if mode],
            ["control (already passing)", int(flat["is_control"].sum())],
            ["total", len(flat)],
        ],
    )

    # ---- fold the human review in ---------------------------------------
    try:
        labels = load("ls06_annotations.parquet")
    except SystemExit:
        labels = None
        console.print(
            "\n[dim]No ls06_annotations.parquet — skipping the human-review merge. "
            "Run poc/ls06_annotations.py to fold reviewer verdicts into the dataset.[/dim]"
        )
    if labels is not None:
        label_col = f"annotation.{ds.ANNOTATION_NAME}.label"
        text_col = f"annotation.{ds.ANNOTATION_NAME}.text"
        by_span = existing_examples(client, name)
        merged = 0
        for _, row in labels.iterrows():
            example_id = by_span.get(str(row["context.span_id"]))
            if not example_id:
                continue  # reviewed a turn that didn't make it into the dataset
            # update_example replaces metadata wholesale rather than merging,
            # so read-merge-write -- sending only the verdict would silently
            # strip source_span_id and break the dedup on the next run.
            current = client.read_example(example_id)
            client.update_example(
                example_id,
                metadata={
                    **(current.metadata or {}),
                    "human_label": str(row[label_col]),
                    "human_reason": str(row.get(text_col, "") or ""),
                },
            )
            merged += 1
        if merged:
            console.print(
                f"\n[green]{merged} examples[/green] now carry a reviewer verdict "
                "as metadata — updating an example writes a new dataset version, "
                "so the merge is itself versioned."
            )
        else:
            console.print(
                "\n[yellow]None of the reviewed runs are in this dataset.[/yellow] "
                "[dim]ls06 reviews the highest-priority turns; ls07 selects on eval "
                "verdicts, so the two sets overlap but do not coincide.[/dim]"
            )

    look_at_ls(
        f"Datasets & Experiments → {name}. Splits separate failures from controls.",
        "Open a reviewed example: `human_label` sits in its metadata, carried into "
        "every experiment that reads the dataset.",
        "The Versions tab — every create/update wrote one, and an experiment can "
        "pin `as_of` to any of them. Arize needed --new-version for that; here it "
        "is the default.",
        "[bold]Open the dataset in the Playground[/bold] (UI only): edit the system "
        "prompt against these exact inputs — the manual version of what ls08 automates.",
    )
    done("poc/ls08_experiments.py — run v1 vs v2 against this dataset")


if __name__ == "__main__":
    app()

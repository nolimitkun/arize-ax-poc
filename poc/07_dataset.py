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

The human review from step 06 is then folded back in: the reviewed examples get
the reviewer's verdict as a field *and* as a dataset annotation. That is what
makes this a golden dataset rather than an export -- the labels a human produced
in step 06 would otherwise stay on the spans and never reach the artefact the
experiments actually run against.

Docs: https://arize.com/docs/ax/improve/build-a-dataset
      https://arize.com/docs/ax/evaluate/human-review
"""

from __future__ import annotations

import pandas as pd
import typer

from _common import arize_client, console, done, header, load, look_at, save, table

app = typer.Typer(add_completion=False)

DATASET_NAME = "copilot-failures"

# All eval columns, mapped to the failure they represent. Score < 1.0 is a
# failure for each (they are all maximize-direction). Shared with poc/04b, which
# has to ask the same question about the same turns.
from copilot.evals import TURN_FAILURE_EVALS as FAILING_EVALS  # noqa: E402

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


ANNOTATION_NAME = "human_groundedness"

# Server-enforced ceiling on both update_examples and annotate_examples.
BATCH_LIMIT = 1000


def chunked(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def existing_examples(client, space: str, name: str) -> dict[str, str]:
    """`source_span_id` -> example id, for the examples already in the dataset.

    The ids are assigned server-side, so the only way back to "which example is
    this turn" is the `source_span_id` field written at creation time.
    """
    response = client.datasets.list_examples(dataset=name, space=space, all=True)
    index: dict[str, str] = {}
    for _key, value in response:
        if not isinstance(value, list):
            continue
        for example in value:
            props = getattr(example, "additional_properties", None) or {}
            span = str(props.get("source_span_id", ""))
            if span:
                index[span] = str(example.id)
    return index


def existing_annotation_counts(client, space: str, name: str) -> list[str]:
    """Example ids that read back carrying at least one annotation."""
    response = client.datasets.list_examples(dataset=name, space=space, all=True)
    found = []
    for _key, value in response:
        if not isinstance(value, list):
            continue
        found.extend(str(e.id) for e in value if getattr(e, "annotations", None))
    return found


def promote_human_labels(client, settings, name: str, new_version: str) -> None:
    """Write step 06's human verdicts back onto the dataset examples.

    Two calls, because they do different things and the docs treat them as
    different features. `update_examples` adds fields to the example itself, so
    the verdict travels with the row into every experiment that reads it.
    `annotate_examples` records it as an annotation keyed by annotation config,
    which is what the Datasets UI shows as human feedback and what can be
    overwritten by a later review without rewriting the example.
    """
    from arize._generated.api_client.models import AnnotateRecordInput, AnnotationInput

    try:
        labels = load("06_annotations.parquet")
    except SystemExit:
        console.print(
            "\n[dim]No 06_annotations.parquet — skipping the human-review merge. "
            "Run poc/06_annotations.py to fold reviewer verdicts into the dataset.[/dim]"
        )
        return

    label_col = f"annotation.{ANNOTATION_NAME}.label"
    text_col = f"annotation.{ANNOTATION_NAME}.text"
    if label_col not in labels.columns:
        console.print(f"[yellow]{label_col} not present in the annotations.[/yellow]")
        return

    space = settings.arize_space_name
    by_span = existing_examples(client, space, name)
    if not by_span:
        console.print("[yellow]No examples carried a source_span_id — nothing to merge.[/yellow]")
        return

    updates, annotations = [], []
    for _, row in labels.iterrows():
        example_id = by_span.get(str(row["context.span_id"]))
        if not example_id:
            continue  # reviewed a turn that didn't make it into the dataset
        verdict = str(row[label_col])
        reason = str(row.get(text_col, "") or "")
        updates.append(
            {"id": example_id, "human_label": verdict, "human_reason": reason}
        )
        annotations.append(
            AnnotateRecordInput(
                record_id=example_id,
                values=[
                    # No `score`. The config created in step 06 is categorical,
                    # and the platform derives the score from the label -- send
                    # one anyway and the whole batch 422s with "score must not
                    # be set for categorical configs".
                    AnnotationInput(name=ANNOTATION_NAME, label=verdict, text=reason[:1000])
                ],
            )
        )

    if not updates:
        console.print(
            "\n[yellow]None of the reviewed spans are in this dataset.[/yellow] "
            "[dim]Step 06 reviews the highest-priority turns; step 07 selects on eval "
            "verdicts, so the two sets overlap but do not coincide.[/dim]"
        )
        return

    console.print(f"\nFolding [bold]{len(updates)}[/bold] human verdicts back into the dataset…")
    # Both endpoints cap a request at 1000 records. This POC never gets close,
    # but a tour run against real traffic would -- and the failure is a 4xx
    # partway through, leaving the dataset half-updated with no record of where
    # it stopped.
    for batch in chunked(updates, BATCH_LIMIT):
        client.datasets.update_examples(
            dataset=name,
            space=space,
            examples=batch,
            **({"new_version": new_version} if new_version else {}),
        )
    reviewed = pd.Series([u["human_label"] for u in updates]).value_counts()
    console.print(
        f"[green]{len(updates)} examples[/green] now carry a reviewer verdict as a field "
        f"({', '.join(f'{k}={v}' for k, v in reviewed.items())})"
        + (f", written as dataset version [bold]{new_version}[/bold]." if new_version else ".")
    )

    for batch in chunked(annotations, BATCH_LIMIT):
        client.datasets.annotate_examples(dataset=name, space=space, annotations=batch)
    # The call is accepted and raises nothing, but the annotations do not come
    # back through `list_examples` -- checked repeatedly over several minutes.
    # Either the write is not landing or the read path does not surface it, and
    # from here the two are indistinguishable. Reporting "wrote N annotations"
    # on the strength of a clean return would be the exact failure this POC
    # keeps finding, so the step says what it can actually confirm.
    visible = sum(1 for eid in existing_annotation_counts(client, space, name) if eid)
    if visible:
        console.print(f"[green]{visible} of them are readable back as annotations.[/green]")
    else:
        console.print(
            f"[yellow]The same {len(annotations)} verdicts were also written as dataset "
            "annotations, but none read back through the SDK.[/yellow] "
            "[dim]annotate_examples() returns cleanly; `list_examples` reports no "
            "annotations minutes later. Check the Datasets UI before relying on it — "
            "the `human_label` field above is the part this step can vouch for.[/dim]"
        )


@app.command()
def main(
    name: str = typer.Option(DATASET_NAME, help="Dataset name in Arize"),
    controls: int = typer.Option(12, help="Passing turns to include as a control group"),
    new_version: str = typer.Option(
        "", help="Name a new dataset version for the human-review merge (default: in place)"
    ),
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
        console.print(f"[yellow]Create failed ({exc}); reconciling with the existing one.[/yellow]")
        # Append only what is genuinely new. Appending the whole batch every time
        # is how this dataset reached 105 examples for 35 turns -- three runs of
        # the same tour, each adding a full duplicate set, silently tripling
        # every experiment that reads it.
        known = existing_examples(client, settings.arize_space_name, name)
        fresh = [e for e in examples if str(e["source_span_id"]) not in known]
        if fresh:
            client.datasets.append_examples(
                dataset=name,
                space=settings.arize_space_name,
                examples=fresh,
            )
            console.print(
                f"[green]Appended {len(fresh)} new examples[/green] to {name} "
                f"({len(examples) - len(fresh)} were already there)."
            )
        else:
            console.print(
                f"[green]All {len(examples)} examples are already in {name}[/green] "
                "— nothing appended."
            )
        dataset = client.datasets.get(dataset=name, space=settings.arize_space_name)

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

    merge_failed = False
    try:
        promote_human_labels(client, settings, name, new_version)
    except Exception as exc:  # noqa: BLE001 - reported, then exited non-zero below
        merge_failed = True
        console.print(
            f"\n[red]Merging the human review failed:[/red] {type(exc).__name__}: {exc}"
        )

    look_at(
        f"Datasets → {name}. Each example carries the question plus its expected behaviour.",
        "Open an example that was reviewed: `human_label` is a field on the row, "
        "carried into every experiment that reads the dataset.",
        "Check whether the same verdict also shows as an annotation — the write is "
        "accepted but does not read back through the SDK, so the UI is the only way "
        "to tell whether it landed.",
        "Dataset versions — note that appending writes into the *latest* version "
        "rather than creating one. Pass --new-version to cut a new one, which is "
        "what pins an experiment to a fixed set of inputs.",
        "[bold]Open the dataset in Prompt Playground[/bold] (UI only): edit the system "
        "prompt against these exact inputs and see answers change side by side. "
        "That is the manual version of what step 08 automates.",
    )
    done("poc/08_experiments.py — run v1 vs v2 against this dataset")
    if merge_failed:
        # The dataset exists, so the output above is real -- but half the step
        # did not happen, and `make all` has to be able to see that.
        raise SystemExit(1)


if __name__ == "__main__":
    app()

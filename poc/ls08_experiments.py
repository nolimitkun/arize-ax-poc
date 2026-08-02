#!/usr/bin/env python
"""Step ls08 -- Improve: compare variants on the same dataset, same evaluators.

The LangSmith mirror of poc/08, and the acceptance test for the port: if v2
doesn't beat v1 here, on the LangSmith dataset with the LangSmith experiment
runner, the port has demonstrated machinery but not measurement.

Everything that decides the verdict is imported from 08 -- the task builder,
the evaluator logic, the paired-McNemar `compare` -- so the two platforms
disagree only if the platforms themselves behave differently. What is native
here is `client.evaluate()`: it runs the target over the dataset, records an
experiment per arm, and attaches each evaluator's verdict as feedback.

One honest caveat: `evaluate()` traces each row as an experiment run, and the
target *also* emits its own OTel trace (the copilot.turn tree) into the
tracing project -- two views of the same call, linked by nothing but time.
Arize's runner has the same shape (step 08's runs are traced too); it is the
cost of reusing the production engine as the experiment target.

Docs: https://docs.smith.langchain.com/evaluation
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
from itertools import islice
from typing import Any

import typer

from _common import console, done, header
from _ls_common import look_at_ls, ls_client, require_langsmith

exp = import_module("08_experiments")

app = typer.Typer(add_completion=False)


def wrap_evaluator(fn, key: str):
    """An 08 evaluator (output, dataset_row) -> EvaluationResult, as a
    LangSmith (run, example) evaluator returning a feedback dict.

    The row handed to the inner function merges the example's inputs and
    reference outputs, which restores exactly the flat shape 08's `field()`
    reads -- `question` from inputs, `expected_behavior` from the reference.
    """

    def evaluator(run, example) -> dict:
        row = {**(example.outputs or {}), **(example.inputs or {})}
        result = fn(run.outputs or {}, row)
        comment = f"{result.label}: {result.explanation}" if result.label else result.explanation
        return {"key": key, "score": result.score, "comment": comment}

    evaluator.__name__ = key
    return evaluator


def iter_selected(selected, client, dataset: str):
    """Examples to sample from for --dry-run, whichever form `selected` takes."""
    if isinstance(selected, str):
        return client.list_examples(dataset_name=selected)
    return iter(selected)


def select_batch(client, dataset: str, batch: str):
    """The examples to measure: one traffic sample, or every one collected.

    ls07 appends a batch per run, so by default this dataset answers "across
    all the traffic we have kept" -- which silently becomes a different
    question from "did this change help on what we just saw" as batches pile
    up. `evaluate()` takes an example list, so scoping is a filter here; step
    08 needs a whole dataset per batch because Arize's runner takes only a
    dataset name.

    Returns (data, description) where data is the dataset name (whole dataset)
    or a list of examples.
    """
    if batch == "all":
        return dataset, "every batch in the dataset"

    examples = list(client.list_examples(dataset_name=dataset))

    def stamp_of(example) -> str:
        """This example's traffic sample, precise or inferred.

        Examples written before ls07 stamped batches carry no marker, and
        measuring "everything" on such a dataset is the blend this option
        exists to avoid. Their creation date is a coarser but honest stand-in
        -- a tour run lands its whole batch within one day -- so a dataset
        built before this change is still scopeable, just at day resolution.
        """
        stamped = str((example.metadata or {}).get("batch", ""))
        return stamped or f"{str(example.created_at)[:10]} (by date)"

    stamps = sorted({stamp_of(e) for e in examples} - {""})
    if not stamps:
        console.print(
            "[yellow]This dataset has no examples to scope.[/yellow] Measuring "
            "all of it."
        )
        return dataset, "every batch in the dataset"

    wanted = stamps[-1] if batch == "latest" else batch
    picked = [e for e in examples if stamp_of(e) == wanted]
    if not picked:
        console.print(
            f"[red]No examples in batch {wanted!r}.[/red] Known batches: "
            f"{', '.join(stamps)}."
        )
        raise SystemExit(1)
    console.print(
        f"[dim]Measuring batch {wanted} — {len(picked)} of {len(examples)} examples "
        f"({len(stamps)} batches in the dataset). `--batch all` for the blended "
        f"view.[/dim]"
    )
    return picked, f"batch {wanted}"


def to_frame(results) -> Any:
    """ExperimentResults -> the frame shape 08's compare() machinery expects.

    `to_pandas()` puts each evaluator's score in a `feedback.<name>` column and
    the dataset example in `example_id`; renaming to `eval.<name>.score` lets
    `paired_verdict`, `mean_scores` and `compare` run unmodified.
    """
    df = results.to_pandas()
    renames = {
        c: f"eval.{c.removeprefix('feedback.')}.score"
        for c in df.columns
        if c.startswith("feedback.")
    }
    return df.rename(columns=renames)


@app.command()
def main(
    dataset: str = typer.Option("copilot-failures-ls", help="Dataset to run against"),
    batch: str = typer.Option(
        "latest",
        help="Which traffic sample to measure: 'latest', 'all', or a ls07 batch stamp",
    ),
    concurrency: int = typer.Option(4, help="Parallel task executions"),
    baseline: str = typer.Option("v1", help="Baseline prompt version"),
    candidate: str = typer.Option("v2", help="Candidate prompt version"),
    dry_run: bool = typer.Option(False, help="Run on a 10-row sample only"),
    judge_model: str = typer.Option(
        "deepseek-v4-pro", help="Model backing the groundedness judge"
    ),
    compare_model: str = typer.Option(
        "deepseek-v4-flash",
        help="Third arm: the candidate prompt on this model "
        "('' to skip; ignored if it equals the agent model)",
    ),
) -> None:
    settings = header(
        "ls08",
        "Improve: prompt and model variants against one baseline",
        "evaluate() · datasets & experiments",
    )
    require_langsmith(settings, "experiments")

    from copilot.config import AGENT_MODEL
    from copilot.tracing import flush, init_tracing

    init_tracing(settings)
    client = ls_client(settings)
    selected, scope = select_batch(client, dataset, batch)
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M")
    summaries: dict[str, dict[str, float]] = {}
    frames: dict[str, Any] = {}

    evaluators = [
        wrap_evaluator(exp.build_groundedness(settings, judge_model), "groundedness"),
        *[wrap_evaluator(fn, fn.__name__) for fn in exp.EVALUATORS],
    ]
    names = [fn.__name__ for fn in exp.EVALUATORS] + ["groundedness"]

    # Same arm structure as 08: each arm names its own reference so exactly one
    # variable differs in every comparison (the model arm compares against the
    # candidate prompt, not the baseline).
    arms: list[tuple[str, str, str | None, str | None]] = [
        (baseline, baseline, None, None),
        (candidate, candidate, None, baseline),
    ]
    if compare_model and compare_model == AGENT_MODEL:
        console.print(
            f"\n[yellow]Skipping the model arm:[/yellow] --compare-model is "
            f"{compare_model}, which is already the agent model."
        )
    elif compare_model:
        arms.append(
            (f"{candidate}-{compare_model.split('-')[-1]}", candidate, compare_model, candidate)
        )

    for label, version, model, _ in arms:
        on = f" on [bold]{model}[/bold]" if model else ""
        console.print(
            f"\n[bold cyan]Running {label}[/bold cyan]{on} against '{dataset}' ({scope})…"
        )
        task = exp.build_task(settings, version, model)
        data = list(islice(iter_selected(selected, client, dataset), 10)) if dry_run else selected
        results = client.evaluate(
            lambda inputs, _task=task: _task(inputs),
            data=data,
            evaluators=evaluators,
            experiment_prefix=f"copilot-{label}-{stamp}",
            max_concurrency=concurrency,
            metadata={
                "prompt_version": version,
                "agent_model": model or AGENT_MODEL,
                "agent": "nimbus-copilot",
            },
        )
        flush()
        frame = to_frame(results)
        summaries[label] = exp.mean_scores(frame, names)
        frames[label] = frame
        console.print(
            f"[green]{label} complete[/green] "
            f"({len(frame)} rows, experiment {results.experiment_name})"
        )

    for label, _version, _model, reference in arms:
        if reference:
            exp.compare(reference, label, summaries, frames)

    look_at_ls(
        f"Datasets & Experiments → {dataset} → Experiments: the arms sit side by side.",
        "Select two and Compare — LangSmith lays them out row by row on the same "
        "inputs, feedback deltas highlighted.",
        "Sort by the groundedness score and read a row where v1 failed and v2 passed. "
        "The two answers next to each other are the whole argument for the change.",
        "Each experiment row links to its run; the copilot's own OTel trace of the "
        "same call is in the tracing project (see the caveat in this step's header).",
    )
    done("poc/ls09_prompt_hub.py — publish the winning prompt and load it at runtime")


if __name__ == "__main__":
    app()

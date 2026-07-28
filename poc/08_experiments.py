#!/usr/bin/env python
"""Step 08 -- Improve: v1 vs v2 on the same dataset, with the same evaluators.

This is the acceptance test for the whole POC. Same inputs, same graders, one
variable changed (the system prompt and tool descriptions). If v2 doesn't beat
v1 here, the improvement isn't real.

Docs: https://arize.com/docs/ax/improve/set-up-an-experiment
      https://arize.com/docs/ax/improve/experiment-in-code
      https://arize.com/docs/ax/evaluate/run-evals-on-experiments
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import typer

from _common import console, done, header, look_at, table

app = typer.Typer(add_completion=False)

HEDGE_MARKERS = (
    "doesn't cover",
    "does not cover",
    "not documented",
    "isn't documented",
    "no documentation",
    "don't want to guess",
    "not in our documentation",
    "couldn't find",
    "could not find",
    "unable to find",
    "i don't have",
    "i do not have",
)


def field(row: Any, *names: str) -> str:
    """Read a dataset column, tolerating the platform's attribute prefixes."""
    if hasattr(row, "get"):
        for name in names:
            for key in (name, f"attributes.{name}", f"attributes.input.{name}"):
                value = row.get(key)
                if value not in (None, ""):
                    return str(value)
    return ""


def build_task(settings, version: str):
    """Task fn for `experiments.run`: takes a dataset row, returns the answer."""
    from copilot.agent import run_turn

    def task(dataset_row) -> str:
        question = field(dataset_row, "question", "input.value")
        if not question:
            return ""
        return run_turn(
            question,
            settings=settings,
            prompt_version=version,
            tags=["experiment", f"prompt-{version}"],
            extra_metadata={"experiment_variant": version},
        ).answer

    return task


# --- evaluators -------------------------------------------------------------
# Signature is (output, dataset_row); the function name becomes the eval name.


def groundedness(output: str, dataset_row) -> Any:
    """Did it avoid inventing policy on questions the KB doesn't cover?"""
    from arize.experiments import EvaluationResult

    answer = (output or "").lower()
    expected = field(dataset_row, "expected_behavior")

    if expected != "refuse_no_context":
        return EvaluationResult(
            score=1.0, label="not_applicable", explanation="Question is covered by the KB."
        )
    if not answer.strip():
        return EvaluationResult(score=0.0, label="empty", explanation="No answer produced.")
    if any(marker in answer for marker in HEDGE_MARKERS):
        return EvaluationResult(
            score=1.0,
            label="grounded",
            explanation="Correctly declined to state an undocumented policy.",
        )
    return EvaluationResult(
        score=0.0,
        label="hallucinated",
        explanation="Asserted a policy the documentation does not contain.",
    )


def answers_from_context(output: str, dataset_row) -> Any:
    """Did it produce a usable answer where the KB *does* cover the question?"""
    from arize.experiments import EvaluationResult

    expected = field(dataset_row, "expected_behavior")
    answer = (output or "").strip()

    if expected == "refuse_no_context":
        return EvaluationResult(
            score=1.0, label="not_applicable", explanation="Refusal is correct here."
        )
    if len(answer.split()) < 8:
        return EvaluationResult(
            score=0.0, label="unhelpful", explanation="Answer too short to be useful."
        )
    if any(marker in answer.lower() for marker in HEDGE_MARKERS):
        return EvaluationResult(
            score=0.0,
            label="over_refused",
            explanation="Declined a question the documentation actually covers.",
        )
    return EvaluationResult(score=1.0, label="answered", explanation="Substantive answer given.")


def conciseness(output: str, dataset_row) -> Any:
    from arize.experiments import EvaluationResult

    words = len((output or "").split())
    if words == 0:
        return EvaluationResult(score=0.0, label="empty", explanation="No answer.")
    if words > 250:
        return EvaluationResult(
            score=0.0, label="verbose", explanation=f"{words} words, over the 250-word guideline."
        )
    return EvaluationResult(score=1.0, label="concise", explanation=f"{words} words.")


EVALUATORS = [groundedness, answers_from_context, conciseness]


def mean_scores(df) -> dict[str, float]:
    """Pull per-evaluator mean scores out of the results frame."""
    out: dict[str, float] = {}
    for fn in EVALUATORS:
        col = next(
            (
                c
                for c in df.columns
                if c.endswith(".score") and fn.__name__ in c
            ),
            None,
        )
        if col is not None:
            out[fn.__name__] = float(df[col].mean())
    return out


@app.command()
def main(
    dataset: str = typer.Option("copilot-failures", help="Dataset to run against"),
    concurrency: int = typer.Option(4, help="Parallel task executions"),
    baseline: str = typer.Option("v1", help="Baseline prompt version"),
    candidate: str = typer.Option("v2", help="Candidate prompt version"),
    dry_run: bool = typer.Option(False, help="Run on a 10-row sample only"),
) -> None:
    settings = header(
        "08",
        "Improve: baseline vs candidate experiment",
        "set-up-an-experiment · experiment-in-code · run-evals-on-experiments",
    )

    from _common import arize_client
    from copilot.tracing import flush, init_tracing

    init_tracing(settings)
    client = arize_client(settings)
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M")
    summaries: dict[str, dict[str, float]] = {}

    for version in (baseline, candidate):
        console.print(f"\n[bold cyan]Running {version}[/bold cyan] against '{dataset}'…")
        experiment, results = client.experiments.run(
            name=f"copilot-{version}-{stamp}",
            dataset=dataset,
            space=settings.arize_space_name,
            task=build_task(settings, version),
            evaluators=EVALUATORS,
            concurrency=concurrency,
            dry_run=dry_run,
            metadata={"prompt_version": version, "agent": "nimbus-copilot"},
        )
        flush()
        summaries[version] = mean_scores(results)
        console.print(
            f"[green]{version} complete[/green] "
            f"({len(results)} rows, experiment {getattr(experiment, 'id', 'dry-run')})"
        )

    # ---- the comparison --------------------------------------------------
    base, cand = summaries.get(baseline, {}), summaries.get(candidate, {})
    rows, improved, regressed = [], 0, 0
    for name in sorted(set(base) | set(cand)):
        b, c = base.get(name), cand.get(name)
        if b is None or c is None:
            rows.append([name, f"{b:.2f}" if b is not None else "-", f"{c:.2f}" if c is not None else "-", "-"])
            continue
        delta = c - b
        if delta > 0.01:
            improved += 1
            arrow = f"[green]+{delta:.2f} ▲[/green]"
        elif delta < -0.01:
            regressed += 1
            arrow = f"[red]{delta:.2f} ▼[/red]"
        else:
            arrow = "[dim]0.00 =[/dim]"
        rows.append([name, f"{b:.2f}", f"{c:.2f}", arrow])

    console.print()
    table(f"{baseline} vs {candidate}", ["evaluator", baseline, candidate, "delta"], rows)

    console.print()
    if improved and not regressed:
        console.print(
            f"[bold green]{candidate} wins.[/bold green] {improved} evaluator(s) improved, "
            "none regressed — the prompt change is a real improvement, measured on "
            "the same inputs.\n"
        )
    elif improved and regressed:
        console.print(
            f"[bold yellow]Mixed result.[/bold yellow] {improved} improved, "
            f"{regressed} regressed. This is the case the control group exists to "
            "catch: check whether the gain came at the cost of over-refusing.\n"
        )
    else:
        console.print(
            f"[bold red]No improvement.[/bold red] Read the per-row explanations in "
            "the Arize experiment view before changing the prompt again.\n"
        )

    look_at(
        f"Experiments → copilot-{baseline}-{stamp} and copilot-{candidate}-{stamp}.",
        "Select both and compare — Arize lays them out row by row on the same inputs.",
        "Sort by the groundedness score and read a row where v1 failed and v2 passed. "
        "The two answers next to each other are the whole argument for the change.",
        "Each experiment run is itself traced, so you can open the underlying trace "
        "from any row.",
    )
    done("poc/09_prompt_hub.py — publish the winning prompt and load it at runtime")


if __name__ == "__main__":
    app()

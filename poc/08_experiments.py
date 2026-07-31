#!/usr/bin/env python
"""Step 08 -- Improve: compare variants on the same dataset, same evaluators.

This is the acceptance test for the whole POC. Same inputs, same graders, one
variable changed. If the candidate doesn't beat the baseline here, the
improvement isn't real.

Two variables are worth changing, and the machinery is identical for both:

  prompt   v1 -> v2 on the same model   -- did the rewrite help?
  model    pro -> flash on the same v2 prompt -- how much quality does the
           cheaper model actually cost?

The second is the question a prompt-only experiment cannot answer, and it is
usually the one with money attached. Each arm is compared against the run that
differs from it by exactly one thing -- the model arm against the candidate
prompt, not against the baseline -- and both go through the same paired test.

Docs: https://arize.com/docs/ax/improve/set-up-an-experiment
      https://arize.com/docs/ax/improve/experiment-in-code
      https://arize.com/docs/ax/evaluate/run-evals-on-experiments
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import typer

from _common import console, done, header, look_at, mcnemar_p, table

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


def build_task(settings, version: str, model: str | None = None):
    """Task fn for `experiments.run`: a dataset row in, this run's result out.

    Returns a dict rather than the answer string, because a groundedness judge
    has to see the documentation *this* run retrieved. Grading the new answer
    against the baseline's context, or against a fresh search, would be grading
    something that never happened.

    `model` is the second independent variable. It is recorded in the span
    metadata as well as the experiment metadata so a row in the results frame
    can be traced back to the model that produced it.
    """
    from copilot.agent import run_turn

    variant = version if model is None else f"{version}-{model}"

    def task(dataset_row) -> dict:
        question = field(dataset_row, "question", "input.value")
        if not question:
            return {"answer": "", "retrieved_doc_ids": "", "tool_calls": ""}
        result = run_turn(
            question,
            settings=settings,
            prompt_version=version,
            model=model,
            tags=["experiment", f"prompt-{version}"],
            extra_metadata={"experiment_variant": variant},
        )
        return {
            "answer": result.answer,
            "retrieved_doc_ids": ",".join(result.retrieved_doc_ids),
            "tool_calls": ",".join(result.tool_calls),
        }

    return task


def answer_of(output) -> str:
    """The task returns a dict; tolerate a bare string for older experiments."""
    if isinstance(output, dict):
        return output.get("answer") or ""
    return output or ""


def retrieved_of(output) -> list[str]:
    if isinstance(output, dict):
        return [i for i in str(output.get("retrieved_doc_ids", "")).split(",") if i]
    return []


# --- evaluators -------------------------------------------------------------
# Signature is (output, dataset_row); the function name becomes the eval name.


def build_groundedness(settings, judge_model: str):
    """The same LLM judge poc/04 runs, wired as an experiment evaluator.

    This used to be a keyword heuristic that returned `not_applicable` -- score
    1.0 -- for every question whose expected_behavior wasn't
    `refuse_no_context`. That made v1 score 1.00 here while poc/04's judge scored
    the same agent 0.46, so no prompt change could ever move the number and the
    experiment silently measured nothing. Sharing the judge is the only way the
    experiment can speak to what production measurement found.
    """
    from arize.experiments import EvaluationResult

    from copilot.evals import judge_groundedness
    from copilot.kb import context_for_ids

    def groundedness(output, dataset_row) -> Any:
        answer = answer_of(output)
        question = field(dataset_row, "question", "input.value")
        context = context_for_ids(retrieved_of(output))
        label, score, explanation = judge_groundedness(
            question, answer, context, settings=settings, model=judge_model
        )
        return EvaluationResult(score=score, label=label, explanation=explanation)

    return groundedness


def answers_from_context(output: str, dataset_row) -> Any:
    """Did it produce a usable answer where the KB *does* cover the question?

    This is the counter-metric to groundedness, and it is the reason the dataset
    carries controls: telling the agent to admit gaps risks it refusing
    questions the documentation *does* answer. Watch this while pushing
    groundedness up -- a "fix" that tanks this one is just a refusal machine.

    Rows where refusal is the correct behaviour score None, not 1.0. Scoring
    them 1.0 previously pinned the metric at 1.00 for both variants, so it
    could never move and carried no information.
    """
    from arize.experiments import EvaluationResult

    expected = field(dataset_row, "expected_behavior")
    answer = answer_of(output).strip()

    if expected == "refuse_no_context":
        return EvaluationResult(
            score=None,
            label="not_applicable",
            explanation="Refusal is correct here; excluded from the mean.",
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

    words = len(answer_of(output).split())
    if words == 0:
        return EvaluationResult(score=0.0, label="empty", explanation="No answer.")
    if words > 250:
        return EvaluationResult(
            score=0.0, label="verbose", explanation=f"{words} words, over the 250-word guideline."
        )
    return EvaluationResult(score=1.0, label="concise", explanation=f"{words} words.")


# groundedness is built per-run because it needs settings; see main().
EVALUATORS = [answers_from_context, conciseness]


def score_column(df, name: str) -> str | None:
    return next((c for c in df.columns if c.endswith(".score") and name in c), None)


def example_key(df) -> str | None:
    """Column identifying which dataset example a result row came from.

    Needed to pair v1 against v2. The exact name is not documented, so match on
    shape rather than guessing one literal.
    """
    for candidate in ("example_id", "dataset_example_id", "id"):
        for col in df.columns:
            if col == candidate or col.endswith(f".{candidate}"):
                return col
    return None


def paired_verdict(base_df, cand_df, name: str) -> tuple[int, int, float] | None:
    """(only_v1_passed, only_v2_passed, p) for one evaluator, or None if unpairable."""
    key = example_key(base_df)
    col = score_column(base_df, name)
    if key is None or col is None or key not in cand_df.columns:
        return None
    if score_column(cand_df, name) is None:
        return None

    merged = base_df[[key, col]].merge(
        cand_df[[key, score_column(cand_df, name)]], on=key, suffixes=("_base", "_cand")
    )
    if merged.empty:
        return None
    cols = [c for c in merged.columns if c != key]
    base, cand = merged[cols[0]], merged[cols[1]]
    usable = base.notna() & cand.notna()
    # Scores here are 0/1; treat anything short of 1.0 as a failure.
    base_pass, cand_pass = base[usable] >= 1.0, cand[usable] >= 1.0
    only_base = int((base_pass & ~cand_pass).sum())
    only_cand = int((cand_pass & ~base_pass).sum())
    return only_base, only_cand, mcnemar_p(only_base, only_cand)


def mean_scores(df, names: list[str]) -> dict[str, float]:
    """Pull per-evaluator mean scores out of the results frame.

    Skips NaN, so a judge call that failed on one row lowers the sample size
    rather than poisoning the mean.
    """
    out: dict[str, float] = {}
    for name in names:
        col = next(
            (c for c in df.columns if c.endswith(".score") and name in c),
            None,
        )
        if col is not None:
            out[name] = float(df[col].mean(skipna=True))
    return out


def compare(base_key: str, cand_key: str, summaries: dict, frames: dict) -> None:
    """Render one baseline-vs-variant comparison, with its significance test.

    A delta alone is not a result. On 35 rows one flipped row moves a mean by
    0.03, and an LLM judge re-run on identical inputs moves by about that much
    on its own -- so "any positive delta wins" would declare victory on noise.
    Each evaluator is therefore tested pairwise (same inputs, both variants) and
    only counted when the disagreement is unlikely under chance.

    Lifted out of main() so the model arm is held to the identical standard as
    the prompt arm; a second, looser comparison written inline is how a cheaper
    model ends up looking free.
    """
    base, cand = summaries.get(base_key, {}), summaries.get(cand_key, {})
    base_df, cand_df = frames.get(base_key), frames.get(cand_key)
    rows, improved, regressed, inconclusive, unpaired = [], 0, 0, 0, 0

    for name in sorted(set(base) | set(cand)):
        b, c = base.get(name), cand.get(name)
        if b is None or c is None:
            rows.append([name, f"{b:.2f}" if b is not None else "-",
                         f"{c:.2f}" if c is not None else "-", "-", "-"])
            continue

        delta = c - b
        paired = (
            paired_verdict(base_df, cand_df, name)
            if base_df is not None and cand_df is not None
            else None
        )

        if paired is None:
            # The results frame carried no example-key column, so the rows
            # can't be matched up and there is no valid test to run. Falling
            # back to "is the delta big enough" would be worse than saying
            # nothing: magnitude alone carries no information about sample size
            # or how many rows moved, so a 0.10 delta on 5 rows would be
            # announced as a win -- the exact false positive this whole section
            # exists to prevent. Report it as untestable instead.
            evidence = "[yellow]unpaired — untestable[/yellow]"
            significant = False
            unpaired += 1
        else:
            only_base, only_cand, p = paired
            evidence = f"{only_cand}↑ {only_base}↓  p={p:.3f}"
            significant = p < 0.05
            if p >= 0.05 and abs(delta) > 0.005:
                inconclusive += 1

        if not significant:
            arrow = f"[dim]{delta:+.2f} ≈[/dim]"
        elif delta > 0:
            improved += 1
            arrow = f"[green]{delta:+.2f} ▲[/green]"
        else:
            regressed += 1
            arrow = f"[red]{delta:+.2f} ▼[/red]"

        rows.append([name, f"{b:.2f}", f"{c:.2f}", arrow, evidence])

    console.print()
    table(
        f"{base_key} vs {cand_key}",
        ["evaluator", base_key, cand_key, "delta", "rows changed / McNemar"],
        rows,
    )
    console.print(
        "[dim]≈ means the change is not distinguishable from noise at p<0.05. "
        "`3↑ 1↓` = 3 rows the variant fixed, 1 it broke.[/dim]"
    )

    console.print()
    notes = []
    if inconclusive:
        notes.append(
            f"{inconclusive} moved but not significantly — more rows would be needed "
            "to call those."
        )
    if unpaired:
        notes.append(
            f"{unpaired} could not be paired row-by-row, so no test was run on them "
            "— they are counted neither way."
        )
    trailer = (" " + " ".join(notes)) if notes else ""
    if improved and not regressed:
        console.print(
            f"[bold green]{cand_key} wins.[/bold green] {improved} evaluator(s) improved "
            f"significantly, none regressed.{trailer}\n"
        )
    elif improved and regressed:
        console.print(
            f"[bold yellow]Mixed result.[/bold yellow] {improved} improved, "
            f"{regressed} regressed, both significantly. This is the case the control "
            f"group exists to catch: check whether the gain came at the cost of "
            f"over-refusing.{trailer}\n"
        )
    elif regressed:
        console.print(
            f"[bold red]{cand_key} is worse.[/bold red] {regressed} evaluator(s) "
            f"regressed significantly and none improved.{trailer}\n"
        )
    else:
        console.print(
            f"[bold]No measurable difference[/bold] between {base_key} and {cand_key}. "
            f"Nothing moved beyond noise on {len(base_df) if base_df is not None else 0} "
            f"rows.{trailer} Read the per-row explanations in the Arize experiment view "
            "before changing anything again — and note that a bigger dataset raises "
            "what this test can detect.\n"
        )


@app.command()
def main(
    dataset: str = typer.Option("copilot-failures", help="Dataset to run against"),
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
        "08",
        "Improve: prompt and model variants against one baseline",
        "set-up-an-experiment · experiment-in-code · run-evals-on-experiments",
    )

    from _common import arize_client
    from copilot.config import AGENT_MODEL
    from copilot.tracing import flush, init_tracing

    init_tracing(settings)
    client = arize_client(settings)
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M")
    summaries: dict[str, dict[str, float]] = {}
    frames: dict[str, Any] = {}

    # The LLM judge is the same one poc/04 runs over production spans, so the
    # experiment and the production measurement are directly comparable.
    evaluators = [build_groundedness(settings, judge_model), *EVALUATORS]
    names = [fn.__name__ for fn in EVALUATORS] + ["groundedness"]

    # (arm label, prompt version, model override, what to compare it against).
    #
    # Each arm names its own reference so that exactly one variable differs
    # across every comparison. The model arm is measured against the *candidate*
    # prompt, not the baseline: comparing v2-on-flash to v1-on-pro would change
    # the prompt and the model at once and could not attribute the difference to
    # either.
    arms: list[tuple[str, str, str | None, str | None]] = [
        (baseline, baseline, None, None),
        (candidate, candidate, None, baseline),
    ]
    if compare_model and compare_model == AGENT_MODEL:
        # The baseline arms already run AGENT_MODEL, which COPILOT_AGENT_MODEL
        # can point anywhere. Adding an arm that names the same model would run
        # the identical prompt on the identical model and then present the two
        # runs' stochastic difference under the heading "v2 vs v2-flash" -- a
        # model comparison in which the model never changed.
        console.print(
            f"\n[yellow]Skipping the model arm:[/yellow] --compare-model is "
            f"{compare_model}, which is already the agent model. "
            "[dim]Set COPILOT_AGENT_MODEL or --compare-model to something else "
            "for the comparison to vary anything.[/dim]"
        )
    elif compare_model:
        # The label becomes an experiment name, so it stays in the same
        # [a-z0-9-] shape as the other arms rather than carrying an `@`.
        arms.append(
            (f"{candidate}-{compare_model.split('-')[-1]}", candidate, compare_model, candidate)
        )

    for label, version, model, _ in arms:
        on = f" on [bold]{model}[/bold]" if model else ""
        console.print(f"\n[bold cyan]Running {label}[/bold cyan]{on} against '{dataset}'…")
        experiment, results = client.experiments.run(
            name=f"copilot-{label}-{stamp}",
            dataset=dataset,
            space=settings.arize_space_name,
            task=build_task(settings, version, model),
            evaluators=evaluators,
            concurrency=concurrency,
            dry_run=dry_run,
            metadata={
                "prompt_version": version,
                "agent_model": model or AGENT_MODEL,
                "agent": "nimbus-copilot",
            },
        )
        flush()
        summaries[label] = mean_scores(results, names)
        frames[label] = results
        console.print(
            f"[green]{label} complete[/green] "
            f"({len(results)} rows, experiment {getattr(experiment, 'id', 'dry-run')})"
        )

    # ---- the comparisons -------------------------------------------------
    for label, _version, _model, reference in arms:
        if reference:
            compare(reference, label, summaries, frames)

    look_at(
        "Experiments → "
        + ", ".join(f"copilot-{label}-{stamp}" for label, *_ in arms)
        + ".",
        "Select them and compare — Arize lays them out row by row on the same inputs.",
        "Sort by the groundedness score and read a row where v1 failed and v2 passed. "
        "The two answers next to each other are the whole argument for the change.",
        "Each experiment run is itself traced, so you can open the underlying trace "
        "from any row — including which model answered it.",
    )
    done("poc/09_prompt_hub.py — publish the winning prompt and load it at runtime")


if __name__ == "__main__":
    app()

#!/usr/bin/env python
"""Step 06 -- Evaluate: human review, and whether the judge agrees with it.

An LLM judge is only worth trusting once you've measured it against human
labels. This step:

  1. defines an annotation config (the label schema reviewers use)
  2. creates a labelling queue seeded with the failing spans
  3. writes simulated human labels back via `spans.update_annotations()`
  4. reports judge-vs-human agreement

Step 3 is simulated so the tour runs unattended; in practice a human works the
queue in the UI and step 4 is what you actually care about.

Docs: https://arize.com/docs/ax/evaluate/human-review
      https://arize.com/docs/ax/evaluate/labeling-queues
      https://arize.com/docs/ax/evaluate/align-evals-to-human-feedback
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

import pandas as pd
import typer

from _common import arize_client, console, done, header, load, look_at, save, table, window

app = typer.Typer(add_completion=False)

ANNOTATION_NAME = "human_groundedness"

# Fallback only. A queue's annotators must be real users with access to the
# space -- Arize 404s with "Annotator email not found or does not have access"
# otherwise -- so the reviewer is resolved from the account at runtime and this
# placeholder is what gets written into the annotation's `updated_by` when no
# real user can be found.
REVIEWER_FALLBACK = "poc-reviewer@example.com"


def queue_name(project: str) -> str:
    """One review queue per project, not one per space.

    The annotation config above is deliberately space-wide -- it is a label
    schema, and every tour should grade against the same one. A queue is the
    opposite: its records are spans in a single project, so a fixed name is
    wrong the moment there is a second project.

    That is not hypothetical here. A reset in this POC is a project *rename*
    (deleting a project poisons its name -- see the README), so the second tour
    hits a 409 on a queue that still exists and still points at the first
    tour's spans. The failure is quiet in the worst way: labels keep landing on
    the new project's spans and the agreement number stays correct, so nothing
    downstream complains, while the queue a reviewer actually opens shows the
    old traffic.
    """
    return f"Groundedness review ({project})"


def reviewer_email(client) -> str:
    """A real annotator for this account: env override, else the first user.

    Hardcoding an address would make the queue work for exactly one person, so
    the common case (a single-seat account) resolves itself.
    """
    override = os.getenv("POC_REVIEWER_EMAIL")
    if override:
        return override
    try:
        response = client.users.list(limit=50)
    except Exception as exc:  # noqa: BLE001 - keep going with the placeholder
        console.print(f"  [dim]could not list users ({type(exc).__name__}: {exc})[/dim]")
        return REVIEWER_FALLBACK
    for _key, value in response:
        if isinstance(value, list):
            for user in value:
                email = getattr(user, "email", "")
                if email:
                    return str(email)
    return REVIEWER_FALLBACK


NOISE_RATE = 8  # percent of non-refund turns a reviewer flags anyway


def is_noisy(span_id: str, rate: int = NOISE_RATE) -> bool:
    """Deterministically decide whether this span draws a noisy label.

    A digest rather than `hash()`. Python salts string hashes per process, so
    `hash(span_id)` gives a different answer in every interpreter -- the same
    span was labelled `grounded` on one run and `hallucinated` on the next.
    That is not the noise this is modelling. A real reviewer's mistakes are a
    fixed property of the case they misread, and the version here has to be too:
    poc/06b splits these labels into train and holdout and measures a template
    change against them, which means nothing if the labels move underneath it.
    Re-running poc/06 also overwrites the annotations in AX, so the drift was
    being published, not just held locally.
    """
    return int.from_bytes(hashlib.sha256(span_id.encode()).digest()[:8], "big") % 100 < rate


def simulated_human_label(row: pd.Series) -> tuple[str, float, str]:
    """Stand-in for a human reviewer.

    Uses the fixture's ground truth rather than the judge's output, so
    agreement is a real measurement and not a tautology. The deliberate ~8%
    disagreement rate on non-refund turns reflects that human labels are noisy
    too -- perfect agreement in a POC should make you suspicious, not happy.
    """
    expected = str(row.get("expected_behavior", ""))
    answer = str(row.get("answer", ""))
    hedged = any(
        m in answer.lower()
        for m in ("doesn't cover", "does not cover", "don't want to guess", "not documented")
    )
    if expected == "refuse_no_context":
        if hedged:
            return "grounded", 1.0, "Correctly declined to state an undocumented policy."
        return "hallucinated", 0.0, "Stated a refund policy that is not in the documentation."

    # Non-refund turns: mostly grounded, with a little reviewer noise.
    if is_noisy(str(row.get("span_id", ""))):
        return "hallucinated", 0.0, "Reviewer flagged an unsupported detail in the answer."
    return "grounded", 1.0, "Claims trace back to the retrieved documentation."


@app.command()
def main(
    sample: int = typer.Option(20, help="How many spans to put in the review queue"),
    skip_queue: bool = typer.Option(False, help="Skip queue creation, just write labels"),
    label_all: bool = typer.Option(
        True, help="Simulate labels for every turn, not just the queued ones"
    ),
) -> None:
    settings = header(
        "06",
        "Evaluate: human review queue, labels, and judge agreement",
        "human-review · labeling-queues · align-evals-to-human-feedback",
    )

    from arize.annotation_configs.types import (
        CategoricalAnnotationValue,
        OptimizationDirection,
    )

    client = arize_client(settings)
    turns = load("03_turns.parquet")

    # Prioritise failing turns -- reviewer time is the scarce resource, and it is
    # what caps the queue.
    ranked = turns.sort_values("is_failure", ascending=False).head(sample).copy()
    # ...but the *simulated* labels cost nothing, and everything downstream is
    # limited by how many of them there are. poc/06b splits these in half and
    # runs a significance test on the holdout, which cannot resolve anything at
    # n=10. So the queue stays realistic while the labels cover the full set.
    labelled = turns.copy() if label_all else ranked
    console.print(
        f"Selected [bold]{len(ranked)}[/bold] spans for the review queue "
        f"({int(ranked['is_failure'].sum())} already flagged by code checks)."
    )
    if label_all and len(labelled) > len(ranked):
        console.print(
            f"Simulating labels for all [bold]{len(labelled)}[/bold] turns "
            f"[dim](--no-label-all to label only the queued {len(ranked)})[/dim]"
        )
    console.print()

    # ---- 1. Annotation config -------------------------------------------
    try:
        config = client.annotation_configs.create_categorical(
            name=ANNOTATION_NAME,
            space=settings.arize_space_name,
            values=[
                CategoricalAnnotationValue(label="grounded", score=1),
                CategoricalAnnotationValue(label="hallucinated", score=0),
            ],
            optimization_direction=OptimizationDirection.MAXIMIZE,
        )
        console.print(
            f"[green]Created annotation config[/green] {ANNOTATION_NAME} "
            f"({getattr(config, 'id', '?')})"
        )
        config_id = str(getattr(config, "id", ""))
    except Exception as exc:  # noqa: BLE001 - re-runs hit "already exists"
        console.print(f"[yellow]Annotation config not created ({exc}); reusing if present.[/yellow]")
        existing = client.annotation_configs.list(space=settings.arize_space_name, limit=100)
        items = getattr(existing, "data", None) or getattr(existing, "annotation_configs", [])
        match = next((c for c in items if getattr(c, "name", "") == ANNOTATION_NAME), None)
        config_id = str(getattr(match, "id", "")) if match else ""

    # ---- 2. Labelling queue ---------------------------------------------
    reviewer = reviewer_email(client)
    if not skip_queue and config_id:
        start, end = window(48)
        try:
            project = client.projects.get(
                project=settings.arize_project_name, space=settings.arize_space_name
            )
            # The typed input rather than a plain dict, for two reasons: a
            # dict goes through `AnnotationQueueRecordInput.from_dict`, which
            # json.dumps it and dies on the datetimes ("Object of type datetime
            # is not JSON serializable"); and `record_type` is an enum whose
            # only accepted value is upper-case `SPAN`.
            from arize.annotation_queues.types import AnnotationQueueSpanRecordInput

            name = queue_name(settings.arize_project_name)
            queue = client.annotation_queues.create(
                name=name,
                space=settings.arize_space_name,
                annotation_config_ids=[config_id],
                annotator_emails=[reviewer],
                instructions=(
                    "Read the question and the assistant's answer. Mark 'hallucinated' "
                    "if the answer states any Nimbus policy, number or timeframe that "
                    "isn't in the retrieved documentation -- even if it sounds right. "
                    "Declining to answer an undocumented question is 'grounded'."
                ),
                record_sources=[
                    AnnotationQueueSpanRecordInput(
                        record_type="SPAN",
                        project_id=str(project.id),
                        start_time=start,
                        end_time=end,
                        span_ids=[str(s) for s in ranked["span_id"].tolist()],
                    )
                ],
            )
            console.print(
                f"[green]Created review queue[/green] {name} "
                f"({getattr(queue, 'id', '?')}) with {len(ranked)} records"
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Queue not created: {exc}[/yellow]")
            console.print(
                f"  [dim]No review queue for {settings.arize_project_name}. The labels "
                "below still land on its spans, so the agreement number stands — but "
                "there is nothing for a reviewer to open.[/dim]"
            )

    # ---- 3. Write human labels ------------------------------------------
    console.print("\n[bold]Writing (simulated) human labels[/bold]")
    labels = labelled.apply(simulated_human_label, axis=1, result_type="expand")
    now = datetime.now(timezone.utc)
    annotations = pd.DataFrame(
        {
            "context.span_id": labelled["span_id"].astype(str).values,
            f"annotation.{ANNOTATION_NAME}.label": labels[0].values,
            f"annotation.{ANNOTATION_NAME}.score": labels[1].astype(float).values,
            f"annotation.{ANNOTATION_NAME}.text": labels[2].values,
            f"annotation.{ANNOTATION_NAME}.updated_by": reviewer,
            f"annotation.{ANNOTATION_NAME}.updated_at": int(now.timestamp() * 1000),
        }
    )
    client.spans.update_annotations(
        space_id=settings.arize_space_id,
        project_name=settings.arize_project_name,
        dataframe=annotations,
    )
    console.print(f"[green]Logged {len(annotations)} human annotations.[/green]")
    save("06_annotations.parquet", annotations)

    # ---- 4. Judge vs human ----------------------------------------------
    try:
        judge = load("04_evals.parquet")
    except SystemExit:
        console.print("[yellow]No 04_evals.parquet — skipping agreement.[/yellow]")
        done()
        return

    judge_col = "eval.groundedness.label"
    if judge_col not in judge.columns:
        console.print(f"[yellow]{judge_col} not present (was 04 run with --skip-llm?).[/yellow]")
        done()
        return

    merged = annotations.merge(
        judge[["context.span_id", judge_col]], on="context.span_id", how="inner"
    )
    human_col = f"annotation.{ANNOTATION_NAME}.label"
    agree = (merged[human_col] == merged[judge_col]).sum()
    n = len(merged)

    both_bad = ((merged[human_col] == "hallucinated") & (merged[judge_col] == "hallucinated")).sum()
    judge_only = ((merged[human_col] == "grounded") & (merged[judge_col] == "hallucinated")).sum()
    human_only = ((merged[human_col] == "hallucinated") & (merged[judge_col] == "grounded")).sum()

    table(
        "Judge vs human (label = hallucinated)",
        ["metric", "value"],
        [
            ["compared spans", n],
            ["agreement", f"{100 * agree / n:.0f}%" if n else "n/a"],
            ["both flagged", both_bad],
            ["judge flagged, human didn't (false positive)", judge_only],
            ["human flagged, judge didn't (false negative)", human_only],
        ],
    )
    if n and agree / n < 0.8:
        console.print(
            "\n[yellow]Agreement below 80%.[/yellow] Per the align-evals guide, "
            "the fix is to tighten the judge template using the disagreeing "
            "examples — not to trust the judge's aggregate score yet.\n"
        )

    look_at(
        "Annotations → the review queue, with instructions and assigned reviewer.",
        "Open a queued span in the UI and label it yourself; it lands on the same span.",
        "A span carrying both `eval.groundedness.*` and `annotation.human_groundedness.*` "
        "— side by side is how you decide whether the judge is trustworthy.",
    )
    done("poc/07_dataset.py — turn the failures into a versioned dataset")


if __name__ == "__main__":
    app()

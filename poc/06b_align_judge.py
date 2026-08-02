#!/usr/bin/env python
"""Step 06b -- Evaluate: align the judge to the human labels it disagreed with.

Step 06 measures judge-vs-human agreement and stops there. That leaves the POC
with a number it cannot act on and a judge it has just shown to be unreliable --
while step 08 goes on to declare a winner using that same judge. This step
closes the loop the align-evals guide describes:

  1. find the cases where the judge and the human disagreed
  2. turn those cases into worked examples inside the judge's own template
  3. re-measure agreement on rows the examples were NOT drawn from
  4. publish the aligned template as a new version of the AX evaluator

Step 3 is what makes this honest. Mining examples from the same rows you then
measure on will show a large improvement every time and mean nothing -- the
examples contain the answers. So the labelled rows are split, and the reported
agreement is on the half the judge has never seen.

Docs: https://arize.com/docs/ax/evaluate/align-evals-to-human-feedback
      https://arize.com/docs/ax/evaluate/create-evaluators
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from _common import (
    OUT_DIR,
    arize_client,
    console,
    done,
    header,
    load,
    look_at,
    mcnemar_p,
    require_arize,
    save,
    table,
)

app = typer.Typer(add_completion=False)

ANNOTATION_NAME = "human_groundedness"
AX_EVALUATOR_NAME = "Groundedness"

# How many worked examples the aligned template carries. Few-shot alignment
# stops paying off quickly and each example is prompt cost on every subsequent
# grading call, so this is capped rather than "use everything in train".
MAX_EXAMPLES = 6
BALLAST = 2


def labelled_rows(
    turns: pd.DataFrame,
    human_name: str = "06_annotations.parquet",
    judge_name: str = "04_evals.parquet",
) -> pd.DataFrame:
    """Turns carrying both a human label and a judge verdict, with their context.

    Everything downstream is a comparison between those two columns, so a row
    missing either is not evidence of anything. Dropped here rather than counted
    as agreement or disagreement.
    """
    from copilot.kb import context_for_ids

    human = load(human_name)
    judge = load(judge_name)

    human_col = f"annotation.{ANNOTATION_NAME}.label"
    judge_col = "eval.groundedness.label"
    for frame, col, where in (
        (human, human_col, "the annotations step"),
        (judge, judge_col, "the offline-evals step"),
    ):
        if col not in frame.columns:
            console.print(f"[red]{col} missing[/red] — re-run {where} first.")
            raise SystemExit(1)

    merged = (
        turns.assign(span=turns["span_id"].astype(str))
        .merge(
            human[["context.span_id", human_col, f"annotation.{ANNOTATION_NAME}.text"]],
            left_on="span",
            right_on="context.span_id",
        )
        .merge(judge[["context.span_id", judge_col]], on="context.span_id")
        .rename(
            columns={
                human_col: "human_label",
                judge_col: "judge_label",
                f"annotation.{ANNOTATION_NAME}.text": "human_reason",
            }
        )
    )
    merged = merged[merged["human_label"].notna() & merged["judge_label"].notna()]
    merged = merged[merged["judge_label"] != "error"]
    merged["context"] = merged["retrieved_doc_ids"].apply(
        lambda ids: context_for_ids([i for i in str(ids).split(",") if i])
    )
    merged["agreed"] = merged["human_label"] == merged["judge_label"]
    return merged.reset_index(drop=True)


def split(rows: pd.DataFrame, train_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/holdout split, stratified on whether the judge got the row right.

    Stratifying matters more than usual here because disagreements are the
    scarce class -- a plain random split of twenty rows can easily put every
    disagreement on one side, leaving either nothing to learn from or nothing
    that could show an improvement.
    """
    train_parts, holdout_parts = [], []
    for _, group in rows.groupby("agreed", sort=False):
        shuffled = group.sample(frac=1.0, random_state=seed)
        cut = round(len(shuffled) * train_frac)
        train_parts.append(shuffled.iloc[:cut])
        holdout_parts.append(shuffled.iloc[cut:])
    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    holdout = pd.concat(holdout_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, holdout


def round_robin(groups: list[pd.DataFrame], budget: int) -> pd.DataFrame:
    """Take up to `budget` rows, cycling through groups largest-first.

    Taking the first N of a shuffled pool instead would let one error mode take
    every slot by luck. Cycling guarantees the biggest mode gets the most
    examples while a rarer one still gets at least one -- correcting a judge on
    only half of how it is wrong tends to push it further the other way.
    """
    ordered = sorted(groups, key=len, reverse=True)
    taken, depth = [], 0
    while len(taken) < budget and any(depth < len(g) for g in ordered):
        for group in ordered:
            if depth < len(group) and len(taken) < budget:
                taken.append(group.iloc[[depth]])
        depth += 1
    if taken:
        return pd.concat(taken)
    return groups[0].head(0) if groups else pd.DataFrame()


def worked_examples(train: pd.DataFrame, cap: int = MAX_EXAMPLES) -> list[dict[str, str]]:
    """Disagreements first, then agreements, up to the cap.

    Disagreements are where the template is demonstrably wrong, so they carry
    the signal. But they are not one failure -- a judge that over-flags and a
    judge that under-flags need opposite corrections, and the two get separate
    budgets here in proportion to how often each actually happens.

    Agreements come after them as ballast, one of each label where possible: a
    template shown nothing but corrections learns "flip the obvious answer",
    which trades one bias for its mirror image.
    """
    disagreed = train[~train["agreed"]]
    agreed = train[train["agreed"]]

    ballast_groups = [g for _, g in agreed.groupby("human_label", sort=False)]
    ballast = round_robin(ballast_groups, min(BALLAST, len(agreed)))

    error_modes = [g for _, g in disagreed.groupby(["human_label", "judge_label"], sort=False)]
    corrections = round_robin(error_modes, cap - len(ballast))

    chosen = pd.concat([corrections, ballast])

    return [
        {
            "question": str(row["question"]),
            # Truncated: the full corpus text would dominate the prompt, and the
            # verdict turns on whether one *specific* claim appears in it.
            "context": str(row["context"])[:600],
            "answer": str(row["answer"])[:600],
            "label": str(row["human_label"]),
            "reason": str(row["human_reason"] or "Reviewer's verdict."),
        }
        for _, row in chosen.iterrows()
    ]


def grade(rows: pd.DataFrame, template: str, settings, model: str) -> pd.Series:
    from copilot.evals import judge_groundedness

    return rows.apply(
        lambda r: judge_groundedness(
            str(r["question"]),
            str(r["answer"]),
            str(r["context"]),
            settings=settings,
            model=model,
            template=template,
        )[0],
        axis=1,
    )


CONTEXT_HINTS = ("context", "document", "reference", "retriev", "source")


def supplies_context(template: str) -> bool:
    """Does this template actually hand the judge the source documents?

    Placeholders only. Searching the prose for "retrieved" finds the hosted
    template's own sentence about documentation it retrieved and concludes it
    has the documents -- when its only two placeholders are `{input.value}` and
    `{output.value}`, and it never sees them.
    """
    import re

    placeholders = re.findall(r"\{([a-zA-Z0-9_.]+)\}", template)
    return any(hint in name.lower() for name in placeholders for hint in CONTEXT_HINTS)


def publish(client, settings, examples: list[dict[str, str]], commit: str) -> None:
    """Add the aligned template as a new version of the AX evaluator.

    The hosted evaluator's template is *not* the local one -- it addresses
    `{input.value}` / `{output.value}` rather than the local judge's named
    placeholders. So the worked examples are spliced into whatever AX currently
    holds, read back from the platform, rather than into the local template.
    That also means an edit someone made in the UI is carried forward instead of
    being silently overwritten.
    """
    from arize.evaluators.types import TemplateConfig

    from copilot.evals import build_aligned_template

    current = client.evaluators.get(evaluator=AX_EVALUATOR_NAME, space=settings.arize_space_name)
    config = current.version.template_config

    # The two judges are not the same judge. The hosted template addresses
    # `{input.value}` / `{output.value}` and has no placeholder for the
    # retrieved documentation at all, so it grades groundedness without ever
    # seeing the source -- which is why step 10 shows the offline judge at 0.47
    # and this one at 0.11 on the identical spans. The worked examples transfer;
    # the holdout number measured above does not, and saying so in the commit
    # message is the difference between a record and a claim.
    grades_blind = not supplies_context(config.template)
    if grades_blind:
        console.print(
            "  [yellow]Note: the hosted template has no retrieved-context "
            "placeholder[/yellow] [dim]— it grades without the source documents, so "
            "the holdout agreement measured above does not describe it.[/dim]"
        )

    client.evaluators.create_template_version(
        evaluator=AX_EVALUATOR_NAME,
        space=settings.arize_space_name,
        commit_message=(
            commit
            + (
                "  Measured with the offline judge template, which sees the retrieved "
                "documents this one does not -- indicative, not a measurement of this "
                "evaluator."
                if grades_blind
                else ""
            )
        ),
        template_config=TemplateConfig(
            name=config.name,
            # escape=False: AX substitutes `{input.value}` itself, so doubling
            # the braces would put a literal brace pair in front of the judge.
            template=build_aligned_template(examples, base=config.template, escape=False),
            classification_choices=config.classification_choices,
            direction=config.direction,
            include_explanations=config.include_explanations,
            use_function_calling_if_available=config.use_function_calling_if_available,
            use_structured_output=getattr(config, "use_structured_output", False),
            llm_config=config.llm_config,
        ),
    )


@app.command()
def main(
    judge_model: str = typer.Option("deepseek-v4-pro", help="Model backing the judge"),
    train_frac: float = typer.Option(0.5, help="Fraction of labelled rows used for examples"),
    seed: int = typer.Option(7, help="Split seed"),
    publish_to_ax: bool = typer.Option(True, help="Version the AX evaluator with the result"),
) -> None:
    settings = header(
        "06b",
        "Evaluate: align the judge to human labels, and prove it on held-out rows",
        "align-evals-to-human-feedback · create-evaluators",
    )
    require_arize(settings, "evaluator alignment")

    from copilot.evals import GROUNDEDNESS_TEMPLATE, build_aligned_template

    rows = labelled_rows(load("03_turns.parquet"))
    if rows.empty:
        console.print(
            "[red]No rows carry both a human label and a judge verdict.[/red] "
            "Run poc/04 and poc/06 against the same traffic first."
        )
        raise SystemExit(1)

    train, holdout = split(rows, train_frac, seed)
    console.print(
        f"[bold]{len(rows)}[/bold] rows have both labels "
        f"({int((~rows['agreed']).sum())} disagreements). "
        f"Split {len(train)} train / {len(holdout)} holdout.\n"
    )
    if holdout.empty:
        console.print(
            "[red]Holdout is empty[/red] — lower --train-frac. Measuring on the "
            "training rows would report an improvement that is really just recall."
        )
        raise SystemExit(1)

    examples = worked_examples(train)
    if not examples:
        console.print(
            "[yellow]No worked examples could be built.[/yellow] The training half is "
            "empty — lower --train-frac, or collect more human labels."
        )
        raise SystemExit(1)

    modes = train[~train["agreed"]].groupby(["judge_label", "human_label"]).size()
    console.print(
        f"Built [bold]{len(examples)}[/bold] worked examples from the training half."
    )
    for (judge_said, human_said), count in modes.items():
        console.print(
            f"  [dim]judge said {judge_said}, human said {human_said}: {count} case(s)[/dim]"
        )
    console.print()

    aligned_template = build_aligned_template(examples)

    console.print(f"Re-grading the [bold]{len(holdout)}[/bold] held-out rows with both templates…")
    holdout = holdout.copy()
    holdout["base_label"] = grade(holdout, GROUNDEDNESS_TEMPLATE, settings, judge_model)
    holdout["aligned_label"] = grade(holdout, aligned_template, settings, judge_model)

    usable = holdout[(holdout["base_label"] != "error") & (holdout["aligned_label"] != "error")]
    n = len(usable)
    if not n:
        console.print("[red]Every held-out row failed to grade.[/red] Check the judge model.")
        raise SystemExit(1)

    base_ok = usable["base_label"] == usable["human_label"]
    aligned_ok = usable["aligned_label"] == usable["human_label"]

    # Paired, because both templates graded the identical rows. The only rows
    # carrying information are those one template got right and the other didn't.
    fixed = int((~base_ok & aligned_ok).sum())
    broken = int((base_ok & ~aligned_ok).sum())
    p = mcnemar_p(broken, fixed)

    table(
        "Agreement with the human labels, on held-out rows",
        ["template", "agrees", "of", "agreement"],
        [
            ["original", int(base_ok.sum()), n, f"{100 * base_ok.mean():.0f}%"],
            ["aligned", int(aligned_ok.sum()), n, f"{100 * aligned_ok.mean():.0f}%"],
        ],
    )
    console.print(
        f"\n[bold]{fixed}[/bold] row(s) the alignment fixed, "
        f"[bold]{broken}[/bold] it broke, p={p:.3f} "
        f"[dim](exact McNemar, paired on the same {n} rows)[/dim]"
    )

    if p < 0.05 and fixed > broken:
        console.print("\n[bold green]The aligned judge agrees with humans more often.[/bold green]")
    elif fixed > broken:
        console.print(
            "\n[yellow]It moved the right way but not beyond chance.[/yellow] "
            "Report that as a lead, not a result."
        )
    elif fixed < broken:
        console.print(
            "\n[bold red]The alignment made agreement worse.[/bold red] The worked "
            "examples are teaching the wrong lesson — read them before trying again."
        )
    else:
        console.print("\n[dim]No change on the held-out rows.[/dim]")

    # Stated whatever the outcome, because it bounds what the outcome can mean.
    # A two-sided exact test cannot reach p<0.05 with fewer than six discordant
    # rows however lopsided they are, so on a holdout this small "not
    # significant" is close to guaranteed and is not evidence against alignment.
    if fixed + broken < 6:
        console.print(
            f"[yellow]Only {fixed + broken} row(s) differ between the two templates."
            "[/yellow] Below six, no split of them can reach p<0.05 — the test cannot "
            "resolve this either way. Raise `poc/06_annotations.py --sample` for more "
            "human labels; that, not the template, is the binding constraint.\n"
        )

    save("06b_alignment.parquet", holdout)
    template_path = Path(OUT_DIR) / "06b_aligned_template.txt"
    template_path.write_text(aligned_template)
    console.print(f"[dim]wrote {template_path.name} ({len(aligned_template)} chars)[/dim]")

    if publish_to_ax:
        console.print(f"\nVersioning the AX evaluator '{AX_EVALUATOR_NAME}'…")
        try:
            publish(
                arize_client(settings),
                settings,
                examples,
                commit=(
                    f"Aligned to {len(examples)} human-reviewed cases mined in poc/06b. "
                    f"Offline holdout agreement {100 * base_ok.mean():.0f}% → "
                    f"{100 * aligned_ok.mean():.0f}% on {n} rows (p={p:.3f})."
                ),
            )
            console.print("[green]Published a new version.[/green]")
        except Exception as exc:  # noqa: BLE001 - the measurement above stands regardless
            console.print(
                f"[yellow]Could not version the evaluator: {type(exc).__name__}: {exc}[/yellow]\n"
                "[dim]Run poc/05_online_evals.py first if the evaluator doesn't exist yet.[/dim]"
            )

    look_at(
        f"Eval Hub → {AX_EVALUATOR_NAME} → Versions. The new one carries the reviewed "
        "cases in its template, and its commit message carries the measurement.",
        "Diff it against the previous version — the change is worked examples, not "
        "rewritten instructions. That is the whole technique.",
        "Whether the continuous task from step 05 picks the new version up on its "
        "own is worth confirming rather than assuming: re-run poc/01_trace.py and "
        "check that the next spans were graded by the version you just published.",
    )
    done(
        "poc/07_dataset.py — build the dataset the experiment runs on",
        "poc/08_experiments.py — now measured by a judge with a known agreement rate",
    )


if __name__ == "__main__":
    app()

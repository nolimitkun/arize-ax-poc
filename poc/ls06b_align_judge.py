#!/usr/bin/env python
"""Step ls06b -- Evaluate: align the judge to human labels, prove it held out.

The LangSmith mirror of poc/06b. The alignment machinery is 06b's own,
imported wholesale -- stratified split, round-robin worked examples, paired
McNemar on the holdout -- because none of it is platform code. What differs is
where the result goes.

The asymmetry worth knowing: Arize hosts *evaluator versions*, so 06b ends by
publishing a new template version of the hosted Groundedness evaluator.
LangSmith has no hosted-evaluator object -- judges live as prompts -- so the
aligned template is published to the LangSmith Prompt Hub instead, with the
measurement in the commit description. Same auditable trail, different noun.

Docs: https://docs.smith.langchain.com/prompt_engineering/how_to_guides/manage_prompts_programatically
"""

from __future__ import annotations

from importlib import import_module

import typer

from _common import OUT_DIR, console, done, header, load, mcnemar_p, save, table
from _ls_common import look_at_ls, ls_client, require_langsmith

align = import_module("06b_align_judge")

app = typer.Typer(add_completion=False)

JUDGE_PROMPT_NAME = "nimbus-groundedness-judge"


@app.command()
def main(
    judge_model: str = typer.Option("deepseek-v4-pro", help="Model backing the judge"),
    train_frac: float = typer.Option(0.5, help="Fraction of labelled rows used for examples"),
    seed: int = typer.Option(7, help="Split seed"),
    publish_to_hub: bool = typer.Option(True, help="Push the aligned template to the Prompt Hub"),
) -> None:
    settings = header(
        "ls06b",
        "Evaluate: align the judge to human labels, and prove it on held-out rows",
        "manage_prompts_programatically · annotation_queues",
    )
    require_langsmith(settings, "evaluator alignment")

    from copilot.evals import GROUNDEDNESS_TEMPLATE, build_aligned_template

    rows = align.labelled_rows(
        load("ls03_turns.parquet"),
        human_name="ls06_annotations.parquet",
        judge_name="ls04_evals.parquet",
    )
    if rows.empty:
        console.print(
            "[red]No rows carry both a human label and a judge verdict.[/red] "
            "Run poc/ls04 and poc/ls06 against the same traffic first."
        )
        raise SystemExit(1)

    train, holdout = align.split(rows, train_frac, seed)
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

    examples = align.worked_examples(train)
    if not examples:
        console.print(
            "[yellow]No worked examples could be built.[/yellow] The training half is "
            "empty — lower --train-frac, or collect more human labels."
        )
        raise SystemExit(1)

    modes = train[~train["agreed"]].groupby(["judge_label", "human_label"]).size()
    console.print(f"Built [bold]{len(examples)}[/bold] worked examples from the training half.")
    for (judge_said, human_said), count in modes.items():
        console.print(
            f"  [dim]judge said {judge_said}, human said {human_said}: {count} case(s)[/dim]"
        )
    console.print()

    aligned_template = build_aligned_template(examples)

    console.print(f"Re-grading the [bold]{len(holdout)}[/bold] held-out rows with both templates…")
    holdout = holdout.copy()
    holdout["base_label"] = align.grade(holdout, GROUNDEDNESS_TEMPLATE, settings, judge_model)
    holdout["aligned_label"] = align.grade(holdout, aligned_template, settings, judge_model)

    usable = holdout[(holdout["base_label"] != "error") & (holdout["aligned_label"] != "error")]
    n = len(usable)
    if not n:
        console.print("[red]Every held-out row failed to grade.[/red] Check the judge model.")
        raise SystemExit(1)

    base_ok = usable["base_label"] == usable["human_label"]
    aligned_ok = usable["aligned_label"] == usable["human_label"]
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

    if fixed + broken < 6:
        console.print(
            f"[yellow]Only {fixed + broken} row(s) differ between the two templates."
            "[/yellow] Below six, no split of them can reach p<0.05 — the test cannot "
            "resolve this either way. Raise `poc/ls06_annotations.py --sample` for more "
            "human labels; that, not the template, is the binding constraint.\n"
        )

    save("ls06b_alignment.parquet", holdout)
    template_path = OUT_DIR / "ls06b_aligned_template.txt"
    template_path.write_text(aligned_template)
    console.print(f"[dim]wrote {template_path.name} ({len(aligned_template)} chars)[/dim]")

    if publish_to_hub:
        console.print(f"\nPushing the aligned template to the Prompt Hub as '{JUDGE_PROMPT_NAME}'…")
        try:
            from langchain_core.prompts import PromptTemplate

            url = ls_client(settings).push_prompt(
                JUDGE_PROMPT_NAME,
                object=PromptTemplate.from_template(aligned_template),
                description="Groundedness judge for the Nimbus copilot, aligned to human review.",
                commit_description=(
                    f"Aligned to {len(examples)} human-reviewed cases mined in poc/ls06b. "
                    f"Holdout agreement {100 * base_ok.mean():.0f}% → "
                    f"{100 * aligned_ok.mean():.0f}% on {n} rows (p={p:.3f})."
                ),
            )
            console.print(f"[green]Pushed a new commit.[/green] {url}")
        except Exception as exc:  # noqa: BLE001 - the measurement above stands regardless
            console.print(
                f"[yellow]Could not push the prompt: {type(exc).__name__}: {exc}[/yellow]"
            )

    look_at_ls(
        f"Prompts → {JUDGE_PROMPT_NAME} → Commits. The new one carries the reviewed "
        "cases in its template, and its commit message carries the measurement.",
        "Diff it against the previous commit — the change is worked examples, not "
        "rewritten instructions. That is the whole technique.",
        "Note the asymmetry with Arize: there the aligned template versions a hosted "
        "evaluator; here it versions a prompt, and whatever grades with it must pull "
        "the tagged commit.",
    )
    done("poc/ls07_dataset.py — turn the failures into a versioned dataset")


if __name__ == "__main__":
    app()

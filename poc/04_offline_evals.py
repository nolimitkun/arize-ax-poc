#!/usr/bin/env python
"""Step 04 -- Evaluate: run evaluators over exported spans, log results back.

Two kinds, side by side, which is the distinction the AX evaluator docs draw:

  * LLM-as-a-judge  -- subjective ("is this answer grounded in the context?")
  * code evaluators -- deterministic ("did it call lookup_order when it should?")

Results are written back onto the original spans with
`client.spans.update_evaluations()`, so they become filterable columns in the
trace view rather than living in a notebook.

Docs: https://arize.com/docs/ax/evaluate/run-evals-on-traces
      https://arize.com/docs/ax/get-started/get-started-evaluations
"""

from __future__ import annotations

import pandas as pd
import typer

from _common import arize_client, console, done, header, load, look_at, save, table

app = typer.Typer(add_completion=False)

# Eval names must match ^[a-zA-Z0-9_\s]+$ -- no dots or hyphens.
GROUNDEDNESS = "groundedness"
TOOL_SELECTION = "tool_selection"
ESCALATION = "escalation_appropriate"
CONCISENESS = "conciseness"

# One definition of groundedness, shared with poc/08 -- see copilot/evals.py.
from copilot.evals import CHOICE_DESCRIPTIONS, GROUNDEDNESS_TEMPLATE  # noqa: E402



def build_llm_evaluators(model: str, settings):
    """Phoenix Evals against DeepSeek V4, via its OpenAI-compatible endpoint.

    Two DeepSeek constraints collide with Phoenix's defaults here, and both
    surface as a 400 raised from deep inside the adapter:

      * `response_format: json_schema` is not supported at all, so Phoenix's
        preferred structured-output path fails on every row.
      * its tool-calling fallback pins `tool_choice` to one named function, and
        V4 rejects a forced tool_choice while thinking mode is on -- which it
        is by default. So the fallback fails too, and the row errors out.

    Both are fixed here without patching Phoenix. `ClassificationEvaluator`
    forwards **kwargs to the chat-completions call as invocation parameters,
    which is the seam for turning thinking off (`create_classifier` does not
    accept them, hence constructing the evaluator directly). And giving each
    choice a `(score, description)` tuple makes Phoenix treat the labels as a
    dict, which routes it straight to tool calling instead of burning a failed
    structured-output request first -- while also telling the judge what the
    labels actually mean.
    """
    from phoenix.evals import LLM
    from phoenix.evals.evaluators import ClassificationEvaluator

    from copilot.config import THINKING_OFF

    llm = LLM(
        provider="openai",
        model=model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )
    return [
        ClassificationEvaluator(
            name=GROUNDEDNESS,
            prompt_template=GROUNDEDNESS_TEMPLATE,
            llm=llm,
            choices=CHOICE_DESCRIPTIONS,
            direction="maximize",
            extra_body=THINKING_OFF,
        )
    ]


# --- code evaluators -------------------------------------------------------


def eval_tool_selection(row: pd.Series) -> tuple[str, float, str]:
    expected = {t for t in str(row.get("expected_tools", "")).split(",") if t}
    actual = {t for t in str(row.get("tool_calls", "")).split(",") if t}
    if not expected:
        return "not_applicable", 1.0, "No tool expectation recorded for this question."
    missing = expected - actual
    if missing:
        return (
            "incorrect",
            0.0,
            f"Expected {sorted(expected)}; called {sorted(actual) or 'nothing'}. "
            f"Missing: {sorted(missing)}.",
        )
    return "correct", 1.0, f"Called all expected tools: {sorted(expected)}."


def eval_escalation(row: pd.Series) -> tuple[str, float, str]:
    should = "escalate_ticket" in str(row.get("expected_tools", ""))
    did = "escalate_ticket" in str(row.get("tool_calls", ""))
    if should and not did:
        return "missed", 0.0, "User was blocked or frustrated but no escalation was raised."
    if not should and did:
        return "unnecessary", 0.5, "Escalated a question the documentation already answers."
    return "correct", 1.0, "Escalation behaviour matched the situation."


def eval_conciseness(row: pd.Series, limit: int = 250) -> tuple[str, float, str]:
    words = int(row.get("answer_words", 0) or 0)
    if words > limit:
        return "verbose", 0.0, f"{words} words, over the {limit}-word guideline."
    if words == 0:
        return "empty", 0.0, "Answer was empty."
    return "concise", 1.0, f"{words} words."


CODE_EVALUATORS = {
    TOOL_SELECTION: eval_tool_selection,
    ESCALATION: eval_escalation,
    CONCISENESS: eval_conciseness,
}


# --- reading Phoenix's output ---------------------------------------------
#
# evaluate_dataframe returns two columns per evaluator: `<name>_score`, holding
# a dict of {score, label, explanation}, and `<name>_execution_details`. Both
# begin with the evaluator name, so selecting by prefix picks the details
# column and yields a frame of NaN scores that Arize then rejects for having
# neither a label nor a score. Match exactly.


def parse_judge_output(graded: pd.DataFrame, name: str):
    """(labels, scores, explanations) for one evaluator, or None if absent."""
    col = f"{name}_score"
    if col not in graded.columns:
        return None

    def field(key: str, default=None):
        return graded[col].apply(lambda v: v.get(key, default) if isinstance(v, dict) else default)

    return field("label"), field("score").astype(float), field("explanation", "")


def judge_failures(graded: pd.DataFrame, name: str) -> list[str]:
    """Per-row judge exceptions, which Phoenix records rather than raising."""
    col = f"{name}_execution_details"
    if col not in graded.columns:
        return []
    failed = [d for d in graded[col] if isinstance(d, dict) and d.get("exceptions")]
    if not failed:
        return []
    return [f"{len(failed)}/{len(graded)} rows failed in the judge; first: {failed[0]['exceptions'][0]}"]


@app.command()
def main(
    judge_model: str = typer.Option("deepseek-v4-pro", help="Model backing the LLM judge"),
    skip_llm: bool = typer.Option(False, help="Run only the code evaluators"),
    limit: int = typer.Option(0, help="Evaluate only the first N turns"),
) -> None:
    settings = header(
        "04",
        "Evaluate: LLM-as-judge + code evaluators, logged back to spans",
        "run-evals-on-traces · get-started-evaluations",
    )

    turns = load("03_turns.parquet")
    if limit:
        turns = turns.head(limit)
    console.print(f"Evaluating [bold]{len(turns)}[/bold] agent turns.\n")

    results = pd.DataFrame({"context.span_id": turns["span_id"].astype(str)})
    judge_results = pd.DataFrame()

    # ---- code evaluators ------------------------------------------------
    console.print("[bold]Code evaluators[/bold] (deterministic)")
    for name, fn in CODE_EVALUATORS.items():
        applied = turns.apply(fn, axis=1, result_type="expand")
        results[f"eval.{name}.label"] = applied[0].values
        results[f"eval.{name}.score"] = applied[1].astype(float).values
        results[f"eval.{name}.explanation"] = applied[2].values
        mean = applied[1].astype(float).mean()
        console.print(f"  {name:<24} mean score {mean:.2f}")

    # ---- LLM judge -------------------------------------------------------
    if not skip_llm:
        console.print(f"\n[bold]LLM-as-a-judge[/bold] (Phoenix Evals, {judge_model})")
        from phoenix.evals import evaluate_dataframe

        from copilot.kb import context_for_ids

        judge_input = turns[["question", "answer"]].copy()
        # The judge needs the documentation text, not the ids -- see
        # context_for_ids. Grading groundedness from ids alone is guesswork.
        judge_input["retrieved_context"] = turns["retrieved_doc_ids"].apply(
            lambda ids: context_for_ids([i for i in str(ids).split(",") if i])
        )
        graded = evaluate_dataframe(
            dataframe=judge_input,
            evaluators=build_llm_evaluators(judge_model, settings),
        )

        for note in judge_failures(graded, GROUNDEDNESS):
            console.print(f"  [yellow]{note}[/yellow]")

        parsed = parse_judge_output(graded, GROUNDEDNESS)
        if parsed is None:
            console.print(
                f"[red]Judge produced no '{GROUNDEDNESS}_score' column[/red] "
                f"(got {list(graded.columns)})"
            )
        else:
            labels, scores, expl = parsed
            # A row whose judge call failed has no label and no score.
            # Arize validates per eval name across the whole frame and rejects
            # the batch if any row is missing both -- which is the very error
            # this script exists to avoid -- so the judge results go up as
            # their own frame covering only the rows that succeeded.
            ok = labels.notna() & scores.notna()
            judge_results = pd.DataFrame(
                {
                    "context.span_id": results["context.span_id"][ok.values].values,
                    f"eval.{GROUNDEDNESS}.label": labels[ok].values,
                    f"eval.{GROUNDEDNESS}.score": scores[ok].values,
                    f"eval.{GROUNDEDNESS}.explanation": expl[ok].fillna("").values,
                }
            )
            skipped = int((~ok).sum())
            if skipped:
                console.print(
                    f"  [yellow]{skipped} row(s) had no usable judge verdict and are "
                    f"excluded from the {GROUNDEDNESS} upload.[/yellow]"
                )
            if len(judge_results):
                console.print(
                    f"  {GROUNDEDNESS:<24} mean score "
                    f"{judge_results[f'eval.{GROUNDEDNESS}.score'].mean():.2f} "
                    f"({len(judge_results)} rows)"
                )

    # ---- log back to Arize ----------------------------------------------
    client = arize_client(settings)
    console.print("\nWriting evaluations back onto the spans…")

    def upload(frame: pd.DataFrame, what: str) -> None:
        if not len(frame):
            return
        client.spans.update_evaluations(
            space_id=settings.arize_space_id,
            project_name=settings.arize_project_name,
            dataframe=frame,
        )
        console.print(f"[green]Logged {len(frame)} rows[/green] ({what}).")

    upload(results, "code evaluators")
    upload(judge_results, GROUNDEDNESS)

    # Downstream steps (06's agreement check) read one frame, so merge the two
    # back together. Rows the judge couldn't grade keep NaN here, which is
    # honest locally -- it's only the Arize upload that can't carry them.
    combined = (
        results.merge(judge_results, on="context.span_id", how="left")
        if len(judge_results)
        else results
    )
    save("04_evals.parquet", combined)

    # ---- summary ---------------------------------------------------------
    score_cols = [c for c in combined.columns if c.endswith(".score")]
    table(
        "Evaluation summary",
        ["evaluator", "mean score", "graded", "failing turns"],
        [
            [
                c.removeprefix("eval.").removesuffix(".score"),
                f"{combined[c].mean():.2f}",
                int(combined[c].notna().sum()),
                int((combined[c] < 1.0).sum()),
            ]
            for c in score_cols
        ],
    )

    look_at(
        "Traces → the eval columns are now on each span, sortable and filterable.",
        f"Filter `eval.{GROUNDEDNESS}.label = 'hallucinated'` to isolate the refund answers.",
        f"Filter `eval.{TOOL_SELECTION}.label = 'incorrect'` for the mis-routed order questions.",
        "Open a failing span and read the judge's explanation — that text is what "
        "tells you how to rewrite the prompt.",
    )
    done(
        "poc/05_online_evals.py — make these run continuously, in-platform",
        "poc/06_annotations.py — add human labels and check the judge agrees",
    )


if __name__ == "__main__":
    app()

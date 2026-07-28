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

GROUNDEDNESS_TEMPLATE = """\
You are grading a customer-support answer for GROUNDEDNESS.

The assistant may only assert things supported by the retrieved documentation.
If the documentation does not cover the question, the correct behaviour is to
say so plainly -- NOT to state a plausible-sounding policy.

[Question]
{question}

[Retrieved documentation IDs]
{retrieved_doc_ids}

[Assistant answer]
{answer}

Answer "grounded" if every factual claim about Nimbus is supported by the
retrieved documentation, or if the assistant correctly said the documentation
does not cover it.

Answer "hallucinated" if the assistant stated any specific policy, number,
timeframe, or entitlement that the retrieved documentation does not contain --
even if it sounds reasonable. Inventing a refund window is hallucination.
"""


def build_llm_evaluators(model: str, settings):
    """Phoenix Evals against DeepSeek, via its OpenAI-compatible endpoint.

    Phoenix's `openai` adapter first tries `response_format: json_schema` for
    structured output. DeepSeek rejects that with a 400, and Phoenix falls back
    to tool calling and caches the choice for the rest of the run -- so expect
    one discarded request at startup, then normal throughput.

    The judge runs with V4's default thinking mode on: Phoenix forwards kwargs
    to the client constructor, not to each request, so there's no seam here to
    pass `thinking: disabled` through. Slower per row, better grading.
    """
    from phoenix.evals import LLM, create_classifier

    llm = LLM(
        provider="openai",
        model=model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )
    return [
        create_classifier(
            name=GROUNDEDNESS,
            prompt_template=GROUNDEDNESS_TEMPLATE,
            llm=llm,
            choices={"grounded": 1.0, "hallucinated": 0.0},
            direction="maximize",
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

        judge_input = turns[["question", "answer", "retrieved_doc_ids"]].copy()
        judge_input["retrieved_doc_ids"] = judge_input["retrieved_doc_ids"].replace(
            "", "(nothing retrieved)"
        )
        graded = evaluate_dataframe(
            dataframe=judge_input,
            evaluators=build_llm_evaluators(judge_model, settings),
        )

        col = next((c for c in graded.columns if c.startswith(GROUNDEDNESS)), None)
        if col is None:
            console.print(f"[red]Judge produced no '{GROUNDEDNESS}' column[/red]")
        else:
            scores = graded[col].apply(
                lambda v: v.get("score") if isinstance(v, dict) else None
            )
            labels = graded[col].apply(
                lambda v: v.get("label") if isinstance(v, dict) else None
            )
            expl = graded[col].apply(
                lambda v: v.get("explanation", "") if isinstance(v, dict) else ""
            )
            results[f"eval.{GROUNDEDNESS}.label"] = labels.values
            results[f"eval.{GROUNDEDNESS}.score"] = scores.astype(float).values
            results[f"eval.{GROUNDEDNESS}.explanation"] = expl.values
            console.print(f"  {GROUNDEDNESS:<24} mean score {scores.astype(float).mean():.2f}")

    # ---- log back to Arize ----------------------------------------------
    client = arize_client(settings)
    console.print("\nWriting evaluations back onto the spans…")
    client.spans.update_evaluations(
        space_id=settings.arize_space_id,
        project_name=settings.arize_project_name,
        dataframe=results,
    )
    console.print(f"[green]Logged {len(results)} rows.[/green]")

    save("04_evals.parquet", results)

    # ---- summary ---------------------------------------------------------
    score_cols = [c for c in results.columns if c.endswith(".score")]
    table(
        "Evaluation summary",
        ["evaluator", "mean score", "failing turns"],
        [
            [
                c.removeprefix("eval.").removesuffix(".score"),
                f"{results[c].mean():.2f}",
                int((results[c] < 1.0).sum()),
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

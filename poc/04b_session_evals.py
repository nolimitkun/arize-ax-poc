#!/usr/bin/env python
"""Step 04b -- Evaluate: grade whole conversations, not just turns.

Step 04 grades one answer at a time, which is what most eval tooling does and
what most eval tooling misses. A conversation can be made of turns that are each
individually correct -- grounded, concise, right tool -- and still fail: the
customer asks three times, gets three accurate non-answers, and leaves. No
turn-level evaluator can see that, because the failure is not in any one turn.

AX already groups turns into sessions (poc/01 sets `session.id` on every span).
This step evaluates that grouping: rebuild each transcript, judge the outcome,
and write the verdict back so sessions become filterable by how they ended.

Docs: https://arize.com/docs/ax/observe/tracing/sessions-and-users
      https://arize.com/docs/ax/evaluate/run-evals-on-traces
"""

from __future__ import annotations

import pandas as pd
import typer

from _common import arize_client, console, done, header, load, look_at, save, table

app = typer.Typer(add_completion=False)

SESSION_EVAL = "session_outcome"


def build_transcripts(turns: pd.DataFrame) -> pd.DataFrame:
    """One row per session: the transcript, and the span to hang the verdict on.

    The verdict goes on the session's *last* turn rather than on every span in
    it. Writing it to all of them would make the eval's mean depend on how many
    turns each session happened to take -- a chatty session would count five
    times, a short one once, and the "average session outcome" would silently be
    an average weighted by length. One span per session keeps the mean a mean
    over sessions, and the span it lands on is the turn where the outcome was
    actually decided.

    Turns are sorted by `start_time` rather than trusted in the order step 03
    happened to write them. The export comes back chronologically today, but
    nothing documents that -- and `spans.list()` is explicitly descending -- so
    the wrong order would hand the judge a conversation read backwards and pin
    the verdict to the opening turn instead of the closing one. Both failures
    look like a plausible score.
    """
    rows = []
    if "start_time" not in turns.columns:
        console.print(
            "[yellow]No `start_time` column — re-run poc/03_query_spans.py.[/yellow] "
            "[dim]Falling back to the stored row order, which is not guaranteed "
            "to be chronological.[/dim]"
        )
    for session_id, group in turns.groupby("session_id", sort=False):
        if not str(session_id).strip():
            continue
        if "start_time" in group.columns:
            group = group.sort_values("start_time", kind="stable")
        ordered = group.reset_index(drop=True)
        lines = []
        for i, turn in ordered.iterrows():
            lines.append(f"Customer: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")
            tools = str(turn.get("tool_calls", "") or "")
            if tools:
                lines.append(f"[tools called: {tools}]")
            if i < len(ordered) - 1:
                lines.append("")
        rows.append(
            {
                "session_id": str(session_id),
                "turns": len(ordered),
                "transcript": "\n".join(lines),
                # Last turn: the outcome is only knowable once the conversation
                # has ended.
                "span_id": str(ordered.iloc[-1]["span_id"]),
                "escalated": "escalate_ticket" in ",".join(
                    str(t) for t in ordered["tool_calls"].fillna("")
                ),
                "turn_failures": int(ordered["is_failure"].sum()),
            }
        )
    return pd.DataFrame(rows)


@app.command()
def main(
    judge_model: str = typer.Option("deepseek-v4-pro", help="Model backing the session judge"),
    limit: int = typer.Option(0, help="Grade only the first N sessions"),
    skip_upload: bool = typer.Option(False, help="Grade locally without writing to Arize"),
) -> None:
    settings = header(
        "04b",
        "Evaluate: session-level outcomes over whole conversations",
        "sessions-and-users · run-evals-on-traces",
    )

    from copilot.evals import SESSION_CHOICES, judge_session

    turns = load("03_turns.parquet")
    sessions = build_transcripts(turns)
    if sessions.empty:
        console.print(
            "[red]No sessions found.[/red] Step 03 exported no `session_id` — check "
            "that poc/01_trace.py ran (it is what sets session.id on the spans)."
        )
        raise SystemExit(1)
    if limit:
        sessions = sessions.head(limit)

    console.print(
        f"Grading [bold]{len(sessions)}[/bold] sessions "
        f"({int(sessions['turns'].sum())} turns, "
        f"{sessions['turns'].mean():.1f} per session).\n"
    )

    verdicts = sessions["transcript"].apply(
        lambda t: judge_session(t, settings=settings, model=judge_model)
    )
    sessions["label"] = [v[0] for v in verdicts]
    sessions["score"] = [v[1] for v in verdicts]
    sessions["explanation"] = [v[2] for v in verdicts]

    graded = sessions[sessions["score"].notna()]
    ungraded = len(sessions) - len(graded)
    if ungraded:
        console.print(
            f"[yellow]{ungraded} session(s) produced no usable verdict[/yellow] and are "
            "excluded from the counts below rather than scored as a failure."
        )

    counts = graded["label"].value_counts()
    table(
        "Session outcomes",
        ["outcome", "sessions", "% of graded"],
        [
            [label, count, f"{100 * count / len(graded):.0f}%"]
            for label, count in counts.items()
        ]
        or [["(none graded)", 0, "-"]],
    )

    # The point of the whole step, stated as a number: sessions that failed as a
    # whole while every turn in them passed the turn-level checks.
    silent = graded[(graded["score"] < 1.0) & (graded["turn_failures"] == 0)]
    console.print(
        f"\n[bold]{len(silent)}[/bold] session(s) ended badly with "
        f"[bold]no turn-level failure[/bold] in them — invisible to step 04.\n"
    )
    for _, row in silent.head(3).iterrows():
        console.print(f"  [yellow]{row['label']}[/yellow]  {row['session_id']} "
                      f"({row['turns']} turns)")
        console.print(f"    [dim]{row['explanation'][:120]}[/dim]")

    save("04b_sessions.parquet", sessions)

    if not skip_upload and len(graded):
        client = arize_client(settings)
        console.print("\nWriting session verdicts onto the closing span of each session…")
        payload = pd.DataFrame(
            {
                "context.span_id": graded["span_id"].values,
                f"eval.{SESSION_EVAL}.label": graded["label"].values,
                f"eval.{SESSION_EVAL}.score": graded["score"].astype(float).values,
                f"eval.{SESSION_EVAL}.explanation": graded["explanation"].values,
            }
        )
        client.spans.update_evaluations(
            space_id=settings.arize_space_id,
            project_name=settings.arize_project_name,
            dataframe=payload,
        )
        console.print(f"[green]Logged {len(payload)} session verdicts.[/green]")

    console.print(
        f"\n[dim]Scale: {', '.join(f'{k}={v}' for k, v in SESSION_CHOICES.items())}. "
        "Three outcomes rather than pass/fail — 'couldn't help' and 'made it worse' "
        "are different problems with different fixes.[/dim]"
    )

    look_at(
        "Sessions → the conversation list, now with an outcome per session.",
        f"Traces → filter `eval.{SESSION_EVAL}.label = 'unresolved'`, then open the "
        "session from the span. Each turn looks fine; the conversation doesn't.",
        "Compare a 'frustrated' session against a 'resolved' one of the same length — "
        "the difference is usually one missing escalation.",
    )
    done(
        "poc/05_online_evals.py — make evaluation continuous and in-platform",
        "poc/06_annotations.py — human labels, and whether the judge agrees",
    )


if __name__ == "__main__":
    app()

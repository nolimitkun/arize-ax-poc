#!/usr/bin/env python
"""Step ls04b -- Evaluate: grade whole conversations (LangSmith threads).

The LangSmith mirror of poc/04b, with the same reasoning imported from it:
turn-level evaluators cannot see a conversation that fails as a whole while
every turn in it passes, so transcripts are rebuilt per session and judged as
one outcome.

The grouping key is `metadata.session_id`, which is also what LangSmith's own
Threads view groups on. The OTel ingest does not derive it from the
`session.id` span attribute, so both engines now write it into metadata as
well -- runs traced before that change have no session_id here and are
reported, not silently dropped.

The verdict lands as `session_outcome` feedback on the thread's *last*
`copilot.turn` run -- same placement logic as 04b, same reason: one verdict
per session keeps the mean a mean over sessions.

Docs: https://docs.smith.langchain.com/observability/how_to_guides/threads
"""

from __future__ import annotations

from importlib import import_module

import typer

from _common import console, done, header, load, save, table
from _ls_common import look_at_ls, ls_client, ls_project_id, require_langsmith

sess = import_module("04b_session_evals")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    judge_model: str = typer.Option("deepseek-v4-pro", help="Model backing the session judge"),
    limit: int = typer.Option(0, help="Grade only the first N sessions"),
    skip_upload: bool = typer.Option(False, help="Grade locally without writing feedback"),
) -> None:
    settings = header(
        "ls04b",
        "Evaluate: thread-level outcomes over whole conversations",
        "threads · attach_user_feedback",
    )
    require_langsmith(settings, "thread-level feedback")

    from copilot.evals import SESSION_CHOICES, judge_session

    turns, verdict_sources = sess.with_turn_verdicts(
        load("ls03_turns.parquet"), evals_name="ls04_evals.parquet"
    )
    no_session = int((turns["session_id"].astype(str).str.strip() == "").sum())
    if no_session:
        console.print(
            f"[yellow]{no_session} turn(s) carry no metadata.session_id[/yellow] — traced "
            "before the engines wrote it into metadata. They are excluded here; re-run "
            "poc/01_trace.py for fully grouped traffic.\n"
        )
    sessions = sess.build_transcripts(turns)
    if sessions.empty:
        console.print(
            "[red]No sessions found.[/red] Check that poc/01_trace.py ran *after* "
            "session_id was added to span metadata — LangSmith cannot group without it."
        )
        raise SystemExit(1)
    if limit:
        sessions = sessions.head(limit)

    console.print(
        f"Grading [bold]{len(sessions)}[/bold] threads "
        f"({int(sessions['turns'].sum())} turns, {sessions['turns'].mean():.1f} per thread).\n"
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
            f"[yellow]{ungraded} thread(s) produced no usable verdict[/yellow] and are "
            "excluded from the counts below rather than scored as a failure."
        )

    counts = graded["label"].value_counts()
    table(
        "Thread outcomes",
        ["outcome", "threads", "% of graded"],
        [[label, count, f"{100 * count / len(graded):.0f}%"] for label, count in counts.items()]
        or [["(none graded)", 0, "-"]],
    )

    silent = graded[(graded["score"] < 1.0) & (graded["turn_failures"] == 0)]
    flagged_rate = float(turns["turn_failed"].mean()) if len(turns) else 0.0
    console.print(
        f"\n[bold]{len(silent)}[/bold] thread(s) ended badly with "
        f"[bold]no turn-level failure[/bold] in them — invisible to every "
        f"per-turn evaluator in this tour.\n"
        f"[dim]'No turn-level failure' means clean under "
        f"{' and '.join(verdict_sources)}.[/dim]"
    )
    if flagged_rate > 0.8:
        console.print(
            f"\n[yellow]Read that zero with care.[/yellow] The turn-level signals flag "
            f"[bold]{100 * flagged_rate:.0f}%[/bold] of all turns, so almost no thread "
            "could come out clean whatever happened in it — the same caveat 04b prints, "
            "for the same judge.\n"
        )
    else:
        console.print()
    for _, row in silent.head(3).iterrows():
        console.print(f"  [yellow]{row['label']}[/yellow]  {row['session_id']} "
                      f"({row['turns']} turns)")
        console.print(f"    [dim]{row['explanation'][:120]}[/dim]")

    save("ls04b_threads.parquet", sessions)

    if not skip_upload and len(graded):
        client = ls_client(settings)
        pid = ls_project_id(client, settings.langsmith_project)
        console.print("\nWriting thread verdicts onto the closing run of each thread…")
        for _, row in graded.iterrows():
            client.create_feedback(
                run_id=row["span_id"],
                key=sess.SESSION_EVAL,
                score=float(row["score"]),
                value=str(row["label"]),
                comment=str(row["explanation"]),
                session_id=pid,
            )
        console.print(f"[green]Logged {len(graded)} thread verdicts.[/green]")

    console.print(
        f"\n[dim]Scale: {', '.join(f'{k}={v}' for k, v in SESSION_CHOICES.items())}. "
        "Three outcomes rather than pass/fail — 'couldn't help' and 'made it worse' "
        "are different problems with different fixes.[/dim]"
    )

    look_at_ls(
        "Tracing Projects → the project → Threads tab: each session is one thread, "
        "grouped by metadata.session_id.",
        f"Runs → filter feedback `{sess.SESSION_EVAL}` < 1 and open the run — the "
        "closing turn of a conversation that failed as a whole.",
        "Compare a 'frustrated' thread against a 'resolved' one of the same length — "
        "the difference is usually one missing escalation.",
    )
    done(
        "poc/ls05_online_rules.py — run the judge continuously, in-platform",
        "poc/ls06_annotations.py — human labels via an annotation queue",
    )


if __name__ == "__main__":
    app()

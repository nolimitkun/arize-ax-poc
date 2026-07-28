#!/usr/bin/env python
"""Step 01 -- Instrument: generate traced traffic.

Runs the copilot over the fixed question set, grouped into multi-turn
conversations so the Sessions view has something to show, and ships every span
to Arize AX via OpenInference.

Docs: https://arize.com/docs/ax/get-started/get-started-tracing
      https://arize.com/docs/ax/instrument/set-up-sessions
      https://arize.com/docs/ax/instrument/track-costs
"""

from __future__ import annotations

import random
import uuid

import typer

from _common import console, done, header, look_at, save, table

app = typer.Typer(add_completion=False)

# Personas give sessions a realistic multi-turn shape: a user arrives with a
# topic and asks two or three related things.
PERSONAS = [
    ("u_ana", "evaluating", ["onboarding", "billing", "limits"]),
    ("u_ben", "billing_issue", ["orders", "refunds", "billing"]),
    ("u_cara", "integrating", ["api", "webhooks", "network"]),
    ("u_dev", "blocked", ["escalation", "troubleshooting"]),
    ("u_eli", "security_review", ["security", "residency", "permissions"]),
    ("u_fay", "power_user", ["transforms", "sla", "orders"]),
]


@app.command()
def main(
    conversations: int = typer.Option(12, help="Number of multi-turn sessions to run"),
    turns: int = typer.Option(3, help="Max turns per session"),
    prompt_version: str = typer.Option("", help="Override COPILOT_PROMPT_VERSION"),
    seed: int = typer.Option(7, help="RNG seed, so runs are reproducible"),
) -> None:
    settings = header(
        "01",
        "Instrument: generate traced traffic",
        "get-started-tracing · set-up-sessions · track-costs",
    )

    from copilot.agent import load_questions, run_conversation
    from copilot.tracing import flush, init_tracing

    version = prompt_version or settings.prompt_version
    init_tracing(settings)
    console.print(f"Tracing initialised. Running with prompt [bold]{version}[/bold].\n")

    rng = random.Random(seed)
    questions = load_questions()
    by_topic: dict[str, list[dict]] = {}
    for q in questions:
        by_topic.setdefault(q["topic"], []).append(q)

    rows, results = [], []
    for i in range(conversations):
        user_id, persona, topics = PERSONAS[i % len(PERSONAS)]
        pool = [q for t in topics for q in by_topic.get(t, [])]
        if not pool:
            continue
        picked = rng.sample(pool, min(turns, len(pool)))
        session_id = f"sess-{persona}-{uuid.uuid4().hex[:8]}"

        console.print(f"[dim]({i + 1}/{conversations}) {user_id} · {persona}[/dim]")
        turn_results = run_conversation(
            [q["question"] for q in picked],
            settings=settings,
            prompt_version=version,
            user_id=user_id,
            session_id=session_id,
            extra_metadata={"persona": persona},
        )
        for q, turn in zip(picked, turn_results):
            results.append(turn)
            rows.append(
                {
                    **turn.to_row(),
                    "question_id": q["id"],
                    "expected_behavior": q["expected_behavior"],
                    "expected_tools": q["expected_tools"],
                    "failure_mode": q["failure_mode"] or "",
                    "persona": persona,
                    "user_id": user_id,
                    "prompt_version": version,
                }
            )
            status = "[red]ERR[/red]" if turn.error else "[green]ok[/green]"
            console.print(
                f"    {status} {q['id']} {q['question'][:58]:<58} "
                f"tools={','.join(turn.tool_calls) or '-'}"
            )

    flush()

    import pandas as pd

    df = pd.DataFrame(rows)
    save("01_local_results.parquet", df)

    errors = int(df["error"].notna().sum()) if "error" in df else 0
    table(
        "Traffic summary",
        ["metric", "value"],
        [
            ["conversations", conversations],
            ["turns", len(df)],
            ["errors", errors],
            ["escalations", int(df["escalated"].sum())],
            ["mean latency (ms)", round(df["latency_ms"].mean(), 0)],
            ["total input tokens", int(df["input_tokens"].sum())],
            ["total output tokens", int(df["output_tokens"].sum())],
        ],
    )

    look_at(
        f"Projects → {settings.arize_project_name} → Traces. Open one trace and "
        "expand the tree: AGENT → CHAIN → RETRIEVER / TOOL / LLM.",
        "Sessions tab — each persona is one session with several turns, showing "
        "duration, turn count and token totals.",
        "Any LLM span → token counts, populated from DeepSeek's usage block. Cost "
        "needs a price entry for the model: Space Settings → Model Costs.",
    )
    done("poc/02_customize_traces.py — custom attributes, metadata, masking")


if __name__ == "__main__":
    app()

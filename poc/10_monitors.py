#!/usr/bin/env python
"""Step 10 -- Observe: monitors and dashboards over the eval metrics.

Evaluation only pays off if a regression pages someone. Monitors watch an eval
metric and alert when it drifts; dashboards put the metrics in one place.

Note: monitors and dashboards are **not** in the Python SDK -- they live behind
the GraphQL API (and the UI). This script checks the metrics exist, then prints
the exact mutations, and runs them if you've supplied a GraphQL-capable key.

Docs: https://arize.com/docs/ax/observe/production-monitoring
      https://arize.com/docs/ax/observe/dashboards
      https://arize.com/docs/ax/graphql-reference/apis/monitors-api
      https://arize.com/docs/ax/observe/production-monitoring/alerting-integrations/slack
"""

from __future__ import annotations

import json
import os

import typer

from _common import arize_client, console, done, header, look_at, table, window

app = typer.Typer(add_completion=False)

GRAPHQL_URL = os.getenv("ARIZE_GRAPHQL_URL", "https://app.arize.com/graphql")

CREATE_MONITOR = """
mutation CreateEvalMonitor($input: CreatePerformanceMonitorMutationInput!) {
  createPerformanceMonitor(input: $input) {
    monitor { id name threshold notificationEmails }
  }
}
"""


def monitor_inputs(project_name: str) -> list[dict]:
    """The three monitors worth having on this agent."""
    return [
        {
            "name": "Groundedness drop",
            "modelName": project_name,
            "evaluationMetric": "eval.groundedness.score",
            "operator": "lessThan",
            "threshold": 0.9,
            "why": "Hallucination regressions -- the failure that started this loop.",
        },
        {
            "name": "Escalation misses",
            "modelName": project_name,
            "evaluationMetric": "eval.escalation_appropriate.score",
            "operator": "lessThan",
            "threshold": 0.95,
            "why": "Blocked users silently not reaching a human.",
        },
        {
            "name": "Turn latency p95",
            "modelName": project_name,
            "evaluationMetric": "latencyP95Ms",
            "operator": "greaterThan",
            "threshold": 20000,
            "why": "Agent loops that stall behind slow tool calls.",
        },
    ]


def post_graphql(api_key: str, query: str, variables: dict) -> dict:
    import httpx

    response = httpx.post(
        GRAPHQL_URL,
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={"query": query, "variables": variables},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


@app.command()
def main(
    apply: bool = typer.Option(
        False, help="Actually POST the mutations (needs ARIZE_GRAPHQL_API_KEY)"
    ),
    hours: int = typer.Option(24, help="Window used to confirm eval metrics exist"),
) -> None:
    settings = header(
        "10",
        "Observe: monitors, alerting and dashboards",
        "production-monitoring · dashboards · monitors GraphQL API",
    )

    # ---- 1. Confirm there is something to monitor ------------------------
    client = arize_client(settings)
    start, end = window(hours)
    console.print("Checking that eval metrics are present on recent spans…")
    df = client.spans.export_to_df(
        space_id=settings.arize_space_id,
        project_name=settings.arize_project_name,
        start_time=start,
        end_time=end,
    )
    eval_cols = sorted(c for c in df.columns if ".score" in c and "eval" in c.lower())
    if eval_cols:
        table(
            "Eval metrics available to monitor",
            ["column", "mean", "non-null"],
            [
                [c, f"{df[c].mean():.2f}", int(df[c].notna().sum())]
                for c in eval_cols
            ],
        )
    else:
        console.print(
            "[yellow]No eval score columns found.[/yellow] Run poc/04 (and give "
            "poc/05's continuous tasks a few minutes) before setting thresholds.\n"
        )

    # ---- 2. Monitors ------------------------------------------------------
    monitors = monitor_inputs(settings.arize_project_name)
    table(
        "Monitors to create",
        ["name", "metric", "condition", "why"],
        [
            [m["name"], m["evaluationMetric"], f"{m['operator']} {m['threshold']}", m["why"]]
            for m in monitors
        ],
    )

    graphql_key = os.getenv("ARIZE_GRAPHQL_API_KEY")
    if apply and graphql_key:
        console.print("\nCreating monitors via GraphQL…")
        for m in monitors:
            payload = {k: v for k, v in m.items() if k != "why"}
            try:
                result = post_graphql(graphql_key, CREATE_MONITOR, {"input": payload})
                if result.get("errors"):
                    console.print(f"  [red]{m['name']}: {result['errors'][0].get('message')}[/red]")
                else:
                    console.print(f"  [green]created[/green] {m['name']}")
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [red]{m['name']}: {exc}[/red]")
    else:
        console.print(
            "\n[dim]Not applied. Monitors need a GraphQL-capable key:\n"
            "  export ARIZE_GRAPHQL_API_KEY=...   # Arize → Settings → API Keys (GraphQL)\n"
            "  uv run python poc/10_monitors.py --apply\n"
            "Mutation shape:[/dim]"
        )
        console.print(f"[dim]{CREATE_MONITOR.strip()}[/dim]")
        console.print(
            f"[dim]variables: {json.dumps({'input': {k: v for k, v in monitors[0].items() if k != 'why'}}, indent=2)}[/dim]"
        )

    # ---- 3. Dashboard and alerting ---------------------------------------
    console.print(
        "\n[bold]Dashboard[/bold] (UI, or the Dashboards GraphQL API):\n"
        "  Dashboards → New → add widgets for\n"
        "    • eval.groundedness.score          — time series, grouped by prompt_version\n"
        "    • eval.tool_selection.label        — distribution\n"
        "    • copilot.escalated                — count over time\n"
        "    • LLM token count + cost           — time series\n"
        "    • p50 / p95 turn latency           — time series\n"
        "  Grouping by `metadata.prompt_version` is what makes a rollout visible: "
        "v1 and v2 traffic appear as two lines on one chart.\n"
    )
    console.print(
        "[bold]Alerting[/bold]: attach Slack, PagerDuty, OpsGenie or Teams to a monitor\n"
        "  Arize → Settings → Integrations → Alerting\n"
        "  docs: /observe/production-monitoring/alerting-integrations/slack\n"
    )

    look_at(
        "Monitors → each monitor's evaluation window, threshold and current status.",
        "Dashboards → build the panel above; group by prompt_version.",
        "Trigger one deliberately: re-run poc/01 with --prompt-version v1 after "
        "promoting v2, and watch groundedness fall below the threshold.",
    )
    done("Tour complete — see README.md for the UI-only steps (Playground, Alyx).")


if __name__ == "__main__":
    app()

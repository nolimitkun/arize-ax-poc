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

def graphql_url(settings) -> str:
    """The GraphQL endpoint for this space's region.

    Same trap as the collector endpoint and the platform client: keys are
    region-scoped, so an EU space driven against app.arize.com authenticates
    against the wrong tenant rather than redirecting. The SDK's own host
    convention is `<service>.<region>.arize.com`, and the app host follows it.
    """
    override = os.getenv("ARIZE_GRAPHQL_URL")
    if override:
        return override
    region = getattr(settings, "arize_region", None)
    host = f"app.{region}.arize.com" if region else "app.arize.com"
    return f"https://{host}/graphql"

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


def resolve_metric(metric: str, available: list[str]) -> tuple[str, str]:
    """Point a monitor at the column the evals are actually logged under.

    Eval columns carry the evaluator's name verbatim, so an evaluator created
    in the UI as `Groundedness` logs `eval.Groundedness.score` while poc/04's
    logs `eval.groundedness.score`. A monitor on the wrong casing is the worst
    kind of broken: it is created successfully, shows green forever, and pages
    nobody -- so the mismatch is repaired here and reported.
    """
    if not metric.startswith("eval.") or not available or metric in available:
        return metric, ""
    actual = {c.lower(): c for c in available}.get(metric.lower())
    if actual:
        return actual, f"logged as `{actual}` — monitor repointed"
    return metric, "no data in this window; the monitor cannot fire until this eval runs"


def post_graphql(url: str, api_key: str, query: str, variables: dict) -> dict:
    import httpx

    response = httpx.post(
        url,
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
        # Eval columns are keyed by the evaluator's name verbatim, so an online
        # evaluator named `Groundedness` in the UI and poc/04's `groundedness`
        # are two unrelated metrics that look like one. A monitor can only watch
        # one of them, and the other regresses unwatched.
        by_case: dict[str, list[str]] = {}
        for column in eval_cols:
            by_case.setdefault(column.lower(), []).append(column)
        for variants in by_case.values():
            if len(variants) > 1:
                console.print(
                    f"  [yellow]{' and '.join(variants)} differ only in case and are "
                    "separate metrics.[/yellow] A monitor on one ignores the other — "
                    "rename the evaluator in Arize so they merge.\n"
                )
    else:
        console.print(
            "[yellow]No eval score columns found.[/yellow] Run poc/04 (and give "
            "poc/05's continuous tasks a few minutes) before setting thresholds.\n"
        )

    # ---- 2. Monitors ------------------------------------------------------
    monitors = monitor_inputs(settings.arize_project_name)
    notes = []
    for m in monitors:
        m["evaluationMetric"], note = resolve_metric(m["evaluationMetric"], eval_cols)
        if note:
            notes.append(f"{m['name']}: {note}")
    table(
        "Monitors to create",
        ["name", "metric", "condition", "why"],
        [
            [m["name"], m["evaluationMetric"], f"{m['operator']} {m['threshold']}", m["why"]]
            for m in monitors
        ],
    )
    for note in notes:
        console.print(f"  [yellow]{note}[/yellow]")

    url = graphql_url(settings)
    graphql_key = os.getenv("ARIZE_GRAPHQL_API_KEY")
    if apply and graphql_key:
        console.print(f"\nCreating monitors via GraphQL ({url})…")
        for m in monitors:
            payload = {k: v for k, v in m.items() if k != "why"}
            try:
                result = post_graphql(url, graphql_key, CREATE_MONITOR, {"input": payload})
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
            f"Endpoint for this region: {url}\n"
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

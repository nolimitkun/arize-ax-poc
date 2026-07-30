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

# An eval score is a *dimension* on the tracing model, not a performance metric.
# `createPerformanceMonitor` only accepts the classic-ML PerformanceMetric enum
# (accuracy, auc, rmse, ...) and has no field for an eval column at all, so it
# cannot express "watch eval.groundedness.score". Data-quality monitors take an
# arbitrary dimension plus an aggregation, which is what these are.
CREATE_MONITOR = """
mutation CreateEvalMonitor($input: CreateDataQualityMonitorMutationInput!) {
  createDataQualityMonitor(input: $input) {
    monitor { id name threshold }
  }
}
"""

# Dimension names come from the model's tracing schema, verified by
# introspection against the live space. Eval dimensions are the full column
# name (`eval.<name>.score`), while latency is a built-in span property named
# `latency_ms` -- not `latencyMs`, and not a performance metric.
LATENCY_DIMENSION = "latency_ms"


def monitor_inputs(space_id: str, project_name: str) -> list[dict]:
    """The three monitors worth having on this agent."""
    common = {
        "spaceId": space_id,
        "modelName": project_name,
        "modelEnvironmentName": "tracing",
    }
    return [
        {
            **common,
            "name": "Groundedness drop",
            "dimensionName": "eval.groundedness.score",
            "dimensionCategory": "llmEval",
            "dataQualityMetric": "avg",
            "operator": "lessThan",
            "threshold": 0.9,
            "why": "Hallucination regressions -- the failure that started this loop.",
        },
        {
            **common,
            "name": "Escalation misses",
            "dimensionName": "eval.escalation_appropriate.score",
            "dimensionCategory": "llmEval",
            "dataQualityMetric": "avg",
            "operator": "lessThan",
            "threshold": 0.95,
            "why": "Blocked users silently not reaching a human.",
        },
        {
            **common,
            "name": "Turn latency p95",
            "dimensionName": LATENCY_DIMENSION,
            "dimensionCategory": "spanProperty",
            "dataQualityMetric": "p95",
            "operator": "greaterThan",
            "threshold": 20000,
            "why": "Agent loops that stall behind slow tool calls.",
        },
    ]


def continuous_eval_columns(client, space: str, project: str) -> set[str]:
    """Eval columns that keep being written, as opposed to written once.

    This is the distinction a monitor lives or dies on. poc/04 grades a batch of
    historic spans and stops; poc/05 attaches a *continuous* evaluation task that
    scores new traffic as it arrives. Both land in `eval.<name>.score`, and the
    column tells you nothing about which is which -- so it is read back from the
    tasks themselves: continuous tasks -> their evaluator ids -> those
    evaluators' names.
    """
    try:
        tasks = client.tasks.list(space=space, project=project)
        evaluators = client.evaluators.list(space=space)
    except Exception:  # noqa: BLE001 - listing is a nicety, not the point
        return set()

    def rows(response, *names):
        for attr in names:
            value = getattr(response, attr, None)
            if value:
                return list(value)
        return []

    names_by_id = {
        str(getattr(e, "id", "")): getattr(e, "name", "")
        for e in rows(evaluators, "data", "evaluators")
    }
    live: set[str] = set()
    for task in rows(tasks, "data", "tasks"):
        if not getattr(task, "is_continuous", False):
            continue
        for entry in getattr(task, "evaluators", None) or []:
            ident = str(getattr(entry, "evaluator_id", None) or getattr(entry, "id", ""))
            name = names_by_id.get(ident)
            if name:
                live.add(f"eval.{name}.score")
    return live


def resolve_metric(
    metric: str, available: list[str], continuous: set[str] | None = None
) -> tuple[str, str]:
    """Point a monitor at the column that will still be written tomorrow.

    Two ways to get this wrong, both of which produce a monitor that is created
    successfully, shows green forever, and pages nobody:

      * Wrong casing. Columns carry the evaluator's name verbatim, so poc/05's
        `Groundedness` and poc/04's `groundedness` are unrelated metrics.
      * Right casing, dead column. Preferring an exact spelling picks poc/04's
        one-time batch over poc/05's continuous evaluator, and the monitor stops
        seeing data as those historic spans age out of its window.

    So a continuously-written column wins over an exact name match, and a metric
    with no continuous writer at all is called out rather than quietly created.
    """
    if not metric.startswith("eval."):
        return metric, ""

    live = sorted(c for c in (continuous or set()) if c.lower() == metric.lower())
    if live:
        chosen = live[0]
        if chosen == metric:
            return chosen, ""
        return chosen, f"repointed to `{chosen}`, the continuously-updated column"

    same = sorted(c for c in available if c.lower() == metric.lower())
    if same:
        chosen = same[0]
        note = (
            f"`{chosen}` is written by a one-time batch (poc/04), not a continuous "
            "evaluator — this monitor goes blind once those spans age out. Attach a "
            "continuous task in poc/05 to monitor it for real."
        )
        return chosen, note
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
        False, help="Actually POST the mutations (uses ARIZE_API_KEY)"
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
                    "separate metrics.[/yellow] A monitor on one ignores the other; "
                    "the continuously-scored one is picked below.\n"
                )
    else:
        console.print(
            "[yellow]No eval score columns found.[/yellow] Run poc/04 (and give "
            "poc/05's continuous tasks a few minutes) before setting thresholds.\n"
        )

    # ---- 2. Monitors ------------------------------------------------------
    monitors = monitor_inputs(settings.arize_space_id, settings.arize_project_name)
    live = continuous_eval_columns(
        client, settings.arize_space_name, settings.arize_project_name
    )
    console.print(
        f"Continuously-scored eval columns: {', '.join(sorted(live)) or '[yellow]none[/yellow]'}\n"
    )
    notes = []
    for m in monitors:
        m["dimensionName"], note = resolve_metric(m["dimensionName"], eval_cols, live)
        if note:
            notes.append(f"{m['name']}: {note}")
    table(
        "Monitors to create",
        ["name", "dimension", "condition", "why"],
        [
            [
                m["name"],
                m["dimensionName"],
                f"{m['dataQualityMetric']} {m['operator']} {m['threshold']}",
                m["why"],
            ]
            for m in monitors
        ],
    )
    for note in notes:
        console.print(f"  [yellow]{note}[/yellow]")

    url = graphql_url(settings)
    # The space API key works against GraphQL with the x-api-key header, so a
    # separate key is optional rather than required.
    graphql_key = os.getenv("ARIZE_GRAPHQL_API_KEY") or settings.arize_api_key
    if apply and graphql_key:
        console.print(f"\nCreating monitors via GraphQL ({url})…")
        gated = False
        for m in monitors:
            payload = {k: v for k, v in m.items() if k != "why"}
            try:
                result = post_graphql(url, graphql_key, CREATE_MONITOR, {"input": payload})
                error = (result.get("errors") or [{}])[0].get("message", "")
                if "enterprise" in error.lower():
                    # Reads and introspection are allowed on every plan; only
                    # mutations are gated. Say that once rather than three times.
                    gated = True
                    break
                if error:
                    console.print(f"  [red]{m['name']}: {error}[/red]")
                else:
                    console.print(f"  [green]created[/green] {m['name']}")
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [red]{m['name']}: {exc}[/red]")
        if gated:
            console.print(
                "  [yellow]GraphQL mutations are enterprise-only on this account.[/yellow]\n"
                "  Reads work (that is how the dimension names above were verified), "
                "writes do not.\n"
                "  Create these by hand — [bold]Monitors → New Monitor → Data "
                "Quality[/bold] — using the\n"
                "  dimension, aggregation, operator and threshold in the table. The "
                "settings are\n"
                "  identical; only the transport differs.\n"
            )
    else:
        console.print(
            "\n[dim]Not applied. Re-run with --apply to create these:\n"
            "  uv run python poc/10_monitors.py --apply\n"
            "ARIZE_API_KEY is used unless ARIZE_GRAPHQL_API_KEY is set.\n"
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

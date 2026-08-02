#!/usr/bin/env python
"""Step ls10 -- Observe: a custom dashboard over the tour's metrics.

The LangSmith mirror of poc/10, and the weakest 1:1 of the port -- 10 builds
*monitors* (alert thresholds with an evaluation window) plus a dashboard;
LangSmith's public API surface here is custom dashboards (sections + charts),
with alerting configured in the UI. So this step builds the dashboard half
for real and says so, rather than dressing charts up as monitors.

Like 10, it dry-runs by default: it prints exactly what it would create, and
`--apply` creates it. The SDK has no charts surface, so requests go through
the REST endpoints the UI uses (`/charts/section`, `/charts/create`) via the
SDK client's authenticated session.

Docs: https://docs.smith.langchain.com/observability/how_to_guides/dashboards
"""

from __future__ import annotations

import json

import typer

from _common import console, done, header, table
from _ls_common import look_at_ls, ls_client, require_langsmith

app = typer.Typer(add_completion=False)


def section_name(project: str) -> str:
    # Project-scoped, like every named artefact this tour creates: dashboards
    # are workspace-level, and a fixed name would show the first project's
    # charts to the second project's tour while it prints success.
    return f"Nimbus copilot ({project})"


def build_charts(project_id: str) -> list[dict]:
    """The three questions the tour keeps asking, as chart definitions.

    Metric names come from the server's own enum (the 422 on an unknown metric
    lists them); each series is scoped to this project's session id.
    """
    scope = {"session": [project_id]}
    return [
        {
            "chart_type": "line",
            "title": "Turns and errors",
            "series": [
                {"name": "runs", "metric": "run_count", "filters": {**scope, "filter": 'eq(name, "copilot.turn")'}},
                {"name": "error rate", "metric": "error_rate", "filters": {**scope, "filter": 'eq(name, "copilot.turn")'}},
            ],
        },
        {
            "chart_type": "line",
            "title": "Latency p99 (turns)",
            "series": [
                {"name": "p99", "metric": "latency_p99", "filters": {**scope, "filter": 'eq(name, "copilot.turn")'}},
            ],
        },
        {
            "chart_type": "line",
            "title": "Groundedness (mean feedback)",
            "series": [
                {
                    "name": "groundedness",
                    "metric": "feedback_score_avg",
                    # Feedback metrics name their key as a field, not a filter
                    # clause -- the server 422s "Feedback key is required"
                    # otherwise.
                    "feedback_key": "groundedness",
                    "filters": {**scope, "filter": 'eq(name, "copilot.turn")'},
                },
            ],
        },
    ]


@app.command()
def main(
    apply: bool = typer.Option(False, help="Actually create the dashboard (default: dry run)"),
) -> None:
    settings = header(
        "ls10",
        "Observe: a custom dashboard over runs, latency and feedback",
        "dashboards · charts API",
    )
    require_langsmith(settings, "dashboards")

    client = ls_client(settings)
    project = client.read_project(project_name=settings.langsmith_project)
    name = section_name(settings.langsmith_project)
    charts = build_charts(str(project.id))

    table(
        f"Dashboard section: {name}",
        ["chart", "series", "metric(s)"],
        [
            [
                c["title"],
                len(c["series"]),
                ", ".join(s["metric"] for s in c["series"]),
            ]
            for c in charts
        ],
    )
    console.print(
        "\n[dim]Alerting note: poc/10's monitors carry thresholds and fire alerts; "
        "LangSmith alerts are configured per-project in the UI (project → Alerts) "
        "and have no public API here. This step builds the dashboard half.[/dim]\n"
    )

    if not apply:
        console.print(
            "[yellow]Dry run.[/yellow] Nothing was created. Re-run with "
            "[bold]--apply[/bold] to create the section and charts. Payloads:"
        )
        for chart in charts:
            console.print(f"  [dim]{json.dumps(chart)[:220]}…[/dim]")
        done("Re-run with --apply, or build the same charts in the UI (Dashboards → New)")
        return

    # ---- create, then verify by reading back ----------------------------
    # Reading the dashboard back is POST /charts with a time window (GET is
    # 405, and /charts/<anything> routes to a per-chart endpoint). The window
    # is required but irrelevant to the structure it returns.
    from datetime import datetime, timedelta, timezone

    def read_sections() -> list[dict]:
        now = datetime.now(timezone.utc)
        try:
            response = client.request_with_retries(
                "POST",
                "/charts",
                request_kwargs={
                    "json": {
                        "start_time": (now - timedelta(days=1)).isoformat(),
                        "end_time": now.isoformat(),
                    }
                },
            )
            return response.json().get("sections", [])
        except Exception as exc:  # noqa: BLE001 - reported by the caller
            console.print(f"[yellow]Could not read the dashboard back: {exc}[/yellow]")
            return []

    def post(path: str, payload: dict) -> tuple[dict | None, str]:
        # request_with_retries raises on 4xx rather than returning the
        # response, so the error path is an exception path.
        try:
            response = client.request_with_retries(
                "POST", path, request_kwargs={"json": payload}
            )
            return response.json(), ""
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

    sections = read_sections()
    existing = next((s for s in sections if s.get("title") == name), None)
    if existing:
        section_id = existing["id"]
        have = {c.get("title") for c in existing.get("charts", [])}
        console.print(f"[dim]Section {name!r} already exists ({section_id}); reusing.[/dim]")
    else:
        created_section, error = post("/charts/section", {"title": name})
        if created_section is None:
            console.print(f"[red]Section not created:[/red] {error[:300]}")
            raise SystemExit(1)
        section_id = created_section["id"]
        have = set()
        console.print(f"[green]Created section[/green] {name} ({section_id})")

    for chart in charts:
        if chart["title"] in have:
            console.print(f"  [dim]{chart['title']} already exists; keeping it[/dim]")
            continue
        made, error = post("/charts/create", {**chart, "section_id": section_id})
        if made is None:
            console.print(f"  [yellow]{chart['title']!r} not created:[/yellow] {error[:300]}")
        else:
            console.print(f"  [green]created[/green] {chart['title']}")

    # Verify by reading back, not by trusting the POSTs.
    final = next((s for s in read_sections() if s.get("title") == name), None)
    visible = {c.get("title") for c in (final or {}).get("charts", [])}
    missing = [c["title"] for c in charts if c["title"] not in visible]
    if missing:
        console.print(
            f"\n[yellow]{len(charts) - len(missing)}/{len(charts)} charts visible "
            f"reading back (missing: {', '.join(missing)}).[/yellow] The section "
            "exists but is incomplete — the errors above say why, and the UI "
            "(Dashboards → the section) shows exactly what a viewer would see."
        )
        raise SystemExit(1)
    console.print(f"\n[green]All {len(charts)} charts verified by reading back.[/green]")

    look_at_ls(
        f"Dashboards → {name}: turns/errors, latency, and groundedness over time.",
        "The groundedness line is fed by ls04's feedback and any online rule from "
        "ls05 — the whole tour on one screen.",
        "Project → Alerts (UI): put a threshold on error rate or feedback score — "
        "the monitor half that poc/10 scripts against Arize's GraphQL API.",
    )
    done("The LangSmith tour is complete — compare it side by side with the Arize one")


if __name__ == "__main__":
    app()

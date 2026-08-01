"""Shared helpers for the LangSmith half of the tour (poc/ls*.py).

The ls-steps mirror steps 02b-10 against LangSmith instead of Arize. The local
logic -- judges, code evaluators, statistics, fixtures -- is imported from the
originals; what lives here is only the platform plumbing that replaces
`arize_client` and `spans.export_to_df`.
"""

from __future__ import annotations

from typing import Any, Iterable

from _common import Settings, console


def ls_client(settings: Settings):
    """The LangSmith client, pointed at the org's own region.

    api_url must come from settings: keys are region-scoped and the US host
    answers an EU key's queries with a plain 403 Forbidden, not a redirect --
    the same trap as Arize regions, in different clothes.
    """
    from langsmith import Client

    return Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_base_url)


def require_langsmith(settings: Settings, what: str) -> None:
    """Exit cleanly when a step needs LangSmith and it is disabled.

    The mirror of `_common.require_arize`: under COPILOT_OBSERVABILITY=arize
    the ls-steps step aside with exit 0, so `make ls-all` composes with any
    mode without crashing half-way.
    """
    if settings.langsmith_enabled:
        return
    console.print(
        f"[yellow]Skipped:[/yellow] this step drives LangSmith platform APIs ({what}), "
        f"which COPILOT_OBSERVABILITY={settings.observability} disables.\n"
        "[dim]Re-run with COPILOT_OBSERVABILITY=langsmith (or both) to include it.[/dim]\n"
    )
    raise SystemExit(0)


def ls_project_id(client, name: str) -> str:
    """The tracing project's id ("session" in LangSmith's own vocabulary).

    Feedback wants it: `create_feedback` without a session_id is deprecated,
    and the queue's RunKey inputs require it outright.
    """
    return str(client.read_project(project_name=name).id)


def app_url(settings: Settings) -> str:
    """The web app that matches the API host (eu.api.smith... -> eu.smith...)."""
    return settings.langsmith_base_url.replace("://api.", "://").replace(
        "://eu.api.", "://eu."
    ).rstrip("/")


def look_at_ls(*items: str) -> None:
    """What to go and check in the LangSmith UI after this step."""
    console.print("\n[bold yellow]Look at this in LangSmith:[/bold yellow]")
    for item in items:
        console.print(f"  → {item}")


def turn_runs_to_df(root_runs: Iterable[Any], children_by_trace: dict[str, list[Any]]):
    """One row per `copilot.turn` run, trajectory rebuilt from its child runs.

    Arize's export hands back the `copilot.*` span attributes as columns;
    LangSmith's OTel ingest *drops* unprefixed custom attributes (verified on
    live runs), so the trajectory is reconstructed the LangSmith-native way --
    from the run tree itself:

      * tools called   -> child runs of run_type "tool" (lookup_order,
        escalate_ticket) plus "search_docs" whenever a `kb.search` retriever
        child exists, because search_docs emits no span of its own -- its body
        *is* the kb.search retriever span (see tools.py).
      * retrieved docs -> the `doc_ids` output of those retriever runs.

    Column names match step 03's frame (`span_id` = the run id) so the judge
    preambles and fixtures join identically on either platform.
    """
    import pandas as pd

    rows = []
    for run in root_runs:
        children = children_by_trace.get(str(run.trace_id), [])
        tools: list[str] = []
        docs: list[str] = []
        for child in sorted(children, key=lambda r: r.start_time):
            if child.run_type == "tool":
                tools.append(child.name)
            elif child.run_type == "retriever" and child.name == "kb.search":
                if "search_docs" not in tools:
                    tools.append("search_docs")
                docs.extend((child.outputs or {}).get("doc_ids", []))
        meta = (run.extra or {}).get("metadata", {})
        rows.append(
            {
                "span_id": str(run.id),
                "trace_id": str(run.trace_id),
                "session_id": str(meta.get("session_id", "")),
                "question": str((run.inputs or {}).get("input", "")),
                "answer": str((run.outputs or {}).get("text", "")),
                "tool_calls": ",".join(tools),
                "retrieved_doc_ids": ",".join(dict.fromkeys(docs)),
                "start_time": run.start_time,
                "prompt_version": str(meta.get("prompt_version", "")),
                "turn_index": meta.get("turn_index"),
                "total_tokens": getattr(run, "total_tokens", None),
                "error": run.error or "",
            }
        )
    return pd.DataFrame(rows)

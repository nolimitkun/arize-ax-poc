"""Shared helpers for the LangSmith half of the tour (poc/ls*.py).

The ls-steps mirror steps 02b-10 against LangSmith instead of Arize. The local
logic -- judges, code evaluators, statistics, fixtures -- is imported from the
originals; what lives here is only the platform plumbing that replaces
`arize_client` and `spans.export_to_df`.
"""

from __future__ import annotations

import uuid
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


# Namespace for deterministic feedback ids. Any fixed UUID works; this one is
# constant so the id for a given (run, key) is stable across machines and runs.
_FEEDBACK_NAMESPACE = uuid.UUID("6f2d7a1e-5c3b-4f8a-9d0e-1b2c3d4e5f60")


def feedback_id(run_id: str, key: str) -> uuid.UUID:
    """The one feedback record a given evaluator owns on a given run."""
    return uuid.uuid5(_FEEDBACK_NAMESPACE, f"{run_id}|{key}")


def upsert_feedback(
    client,
    *,
    run_id: str,
    key: str,
    project_id: str,
    score: float | None = None,
    value: Any = None,
    comment: str = "",
    prune: bool = False,
) -> None:
    """Write one verdict per (run, evaluator), replacing any earlier one.

    `create_feedback` mints a new record every call, so re-running a step --
    after a judge changed its mind, or with a different --limit -- leaves a run
    carrying several verdicts for the same key. Nothing errors; the run just
    quietly counts more than once in every feedback average, and a re-run makes
    the metric drift without the traffic changing.

    Passing a deterministic `feedback_id` makes the write an upsert instead
    (verified against the API: two writes with one id leave one record, the
    second value winning), so a step is safe to re-run and the second run
    corrects the first rather than stacking on it.

    `prune` additionally clears strays written *before* this became an upsert.
    It costs a list call per run, so it is opt-in rather than the default.
    """
    if prune:
        try:
            keep = feedback_id(run_id, key)
            for existing in client.list_feedback(run_ids=[run_id], feedback_key=[key]):
                if existing.id != keep:
                    client.delete_feedback(existing.id)
        except Exception:  # noqa: BLE001 - pruning is housekeeping, not the job
            pass
    client.create_feedback(
        run_id=run_id,
        key=key,
        score=score,
        value=value,
        comment=comment,
        session_id=project_id,
        feedback_id=feedback_id(run_id, key),
    )


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
                "user_id": str(meta.get("user_id", "")),
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

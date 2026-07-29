"""Agent tools, each wrapped in a TOOL span.

Two versions of the tool schemas exist. `v1` descriptions are vague on purpose:
the router reaches for `search_docs` when it should call `lookup_order`, and
never calls `escalate_ticket`. That mis-routing is one of the seeded failure
modes this POC exists to catch (`03` finds it, `08` proves it fixed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from .config import DATA_DIR
from .kb import format_context, search_traced
from .tracing import set_output, span


@dataclass
class ToolCallRecord:
    """What the agent actually did, for trajectory evaluation."""

    name: str
    arguments: dict[str, Any]
    ok: bool = True


@dataclass
class ToolContext:
    """Per-turn state threaded through tool execution."""

    calls: list[ToolCallRecord] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_context: str = ""
    escalated: bool = False

    @property
    def tool_names(self) -> list[str]:
        return [c.name for c in self.calls]


def _load_orders() -> dict[str, dict[str, Any]]:
    return json.loads((DATA_DIR / "orders.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------


def search_docs(ctx: ToolContext, query: str) -> str:
    """Search the corpus, accumulating what the turn has been shown.

    The agent may search several times in one turn, and every earlier result
    stays in the conversation -- so the answer can legitimately draw on any of
    them. Recording only the last search would under-report the context and
    make an offline groundedness judge call supported claims hallucinated.
    The return value is just this search, which is what the model asked for.
    """
    hits = search_traced(query)
    context = format_context(hits)

    for chunk, _ in hits:
        if chunk.doc_id not in ctx.retrieved_doc_ids:
            ctx.retrieved_doc_ids.append(chunk.doc_id)
    if context not in ctx.retrieved_context:
        ctx.retrieved_context = (
            f"{ctx.retrieved_context}\n\n---\n\n{context}"
            if ctx.retrieved_context
            else context
        )
    return context


def lookup_order(ctx: ToolContext, order_id: str) -> str:
    with span(
        "lookup_order",
        OpenInferenceSpanKindValues.TOOL,
        input_value={"order_id": order_id},
    ) as sp:
        sp.set_attribute(SpanAttributes.TOOL_NAME, "lookup_order")
        orders = _load_orders()
        record = orders.get(order_id.strip().upper())
        result = (
            json.dumps(record, indent=2)
            if record
            else f"No order found with ID {order_id!r}. Known IDs look like NMB-4471."
        )
        sp.set_attribute("tool.found", record is not None)
        set_output(sp, result)
        return result


def escalate_ticket(ctx: ToolContext, summary: str, severity: str = "normal") -> str:
    """Records an escalation. Deliberately does not send anything."""
    with span(
        "escalate_ticket",
        OpenInferenceSpanKindValues.TOOL,
        input_value={"summary": summary, "severity": severity},
        tags=["escalation"],
    ) as sp:
        sp.set_attribute(SpanAttributes.TOOL_NAME, "escalate_ticket")
        sp.set_attribute("tool.severity", severity)
        ctx.escalated = True
        ticket_id = f"ESC-{abs(hash(summary)) % 90000 + 10000}"
        result = (
            f"Escalated to a human support engineer. Ticket {ticket_id} created "
            f"at severity {severity!r}. A person will respond directly."
        )
        set_output(sp, result)
        return result


IMPLEMENTATIONS: dict[str, Callable[..., str]] = {
    "search_docs": search_docs,
    "lookup_order": lookup_order,
    "escalate_ticket": escalate_ticket,
}


# --------------------------------------------------------------------------
# Schemas -- OpenAI function-calling shape, which is what DeepSeek accepts.
# v1 is deliberately under-specified.
# --------------------------------------------------------------------------


def _tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def description_of(tool: dict[str, Any]) -> str:
    return tool["function"]["description"]


_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query"},
    },
    "required": ["query"],
}

TOOLS_V1: list[dict[str, Any]] = [
    # Vague and over-broad: invites use for order lookups too.
    _tool("search_docs", "Look up information about Nimbus.", _SEARCH_SCHEMA),
    _tool(
        "lookup_order",
        "Get order info.",
        {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    ),
    _tool(
        "escalate_ticket",
        "Escalate to support.",
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "severity": {"type": "string"},
            },
            "required": ["summary"],
        },
    ),
]

TOOLS_V2: list[dict[str, Any]] = [
    _tool(
        "search_docs",
        "Search the Nimbus product documentation for policy, configuration, and "
        "how-to information. Call this for any question about how Nimbus works. "
        "Do NOT use this to look up a specific customer's order, invoice, plan, "
        "or billing status -- use lookup_order for that.",
        _SEARCH_SCHEMA,
    ),
    _tool(
        "lookup_order",
        "Retrieve a specific customer order by its ID (format NMB-nnnn), returning "
        "plan, status, amount charged, billing period, payment method, and notes. "
        "Call this whenever the user mentions an order ID or asks about THEIR "
        "specific charge, invoice, plan status, or payment. The documentation "
        "does not contain per-customer data -- only this tool does.",
        {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order ID, e.g. NMB-4471",
                }
            },
            "required": ["order_id"],
        },
    ),
    _tool(
        "escalate_ticket",
        "Hand the conversation to a human support engineer. Call this when the "
        "user is blocked and the documentation cannot unblock them, is reporting "
        "a possible security or data-integrity incident, is asking for a billing "
        "correction or refund you cannot action, has already tried support "
        "without resolution, or is clearly frustrated and asking for a human. "
        "Prefer escalating over guessing. You may answer AND escalate in the "
        "same turn.",
        {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-sentence summary of the user's problem",
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "urgent for outages, data loss, or security issues",
                },
            },
            "required": ["summary"],
        },
    ),
]


def tools_for(version: str) -> list[dict[str, Any]]:
    return TOOLS_V1 if version == "v1" else TOOLS_V2


def execute(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a tool call, recording it on the context for later evaluation."""
    impl = IMPLEMENTATIONS.get(name)
    if impl is None:
        ctx.calls.append(ToolCallRecord(name=name, arguments=arguments, ok=False))
        return f"Unknown tool {name!r}."
    ctx.calls.append(ToolCallRecord(name=name, arguments=arguments))
    try:
        return impl(ctx, **arguments)
    except TypeError as exc:
        ctx.calls[-1].ok = False
        return f"Invalid arguments for {name}: {exc}"

"""The Nimbus support copilot.

Trace shape produced by one call to `run_turn`:

    AGENT      copilot.turn
    ├─ CHAIN     router.classify        -> LLM  (auto-instrumented)
    ├─ LLM       (answer generation)         (auto-instrumented, may repeat)
    ├─ RETRIEVER kb.search                   (from search_docs)
    ├─ TOOL      lookup_order
    └─ TOOL      escalate_ticket

A manual tool-use loop is used rather than a framework runner, because each tool
body needs its own TOOL span and a per-turn context object threaded through it --
the trajectory record is what poc/04's code evaluators grade.

The model is DeepSeek V4, called through the `openai` SDK against DeepSeek's
OpenAI-compatible endpoint (see config.DEEPSEEK_BASE_URL). Two V4 behaviours
shape the code below: thinking mode is on by default, and inside a tool loop
the model's `reasoning_content` has to be echoed back on every subsequent
request or the API returns 400.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import openai
from openinference.instrumentation import using_attributes
from openinference.semconv.trace import OpenInferenceSpanKindValues

from .config import AGENT_MODEL, ROUTER_MODEL, THINKING_OFF, THINKING_ON, Settings
from .prompts import load_prompt
from .tools import ToolContext, execute, tools_for
from .tracing import set_output, span

MAX_TOOL_ITERATIONS = 6


@dataclass
class TurnResult:
    question: str
    answer: str
    tool_calls: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_context: str = ""
    escalated: bool = False
    intent: str = "unknown"
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    session_id: str = ""
    error: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "tool_calls": self.tool_calls,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "escalated": self.escalated,
            "intent": self.intent,
            "latency_ms": round(self.latency_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "session_id": self.session_id,
        }


_INTENTS = ("how_to", "billing_account", "troubleshooting", "complaint", "other")

_ROUTER_PROMPT = (
    "Classify the user's support message into exactly one intent. "
    f"Reply with only the label, nothing else. Labels: {', '.join(_INTENTS)}."
)


def _client(settings: Settings) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )


def _classify(client: openai.OpenAI, question: str) -> str:
    """Cheap intent classification, wrapped in a CHAIN span.

    Its output isn't load-bearing for the answer -- it exists so the trace has a
    routing step, which is what makes the AGENT/CHAIN/LLM nesting realistic.
    """
    with span(
        "router.classify",
        OpenInferenceSpanKindValues.CHAIN,
        input_value=question,
        metadata={"model": ROUTER_MODEL, "labels": list(_INTENTS)},
    ) as sp:
        try:
            # Thinking off is load-bearing here, not just a saving: it is on by
            # default, and a 32-token budget would be spent entirely on the CoT,
            # leaving no content to parse.
            resp = client.chat.completions.create(
                model=ROUTER_MODEL,
                max_tokens=32,
                messages=[
                    {"role": "system", "content": _ROUTER_PROMPT},
                    {"role": "user", "content": question},
                ],
                extra_body=THINKING_OFF,
            )
            raw = (resp.choices[0].message.content or "").strip().lower()
            intent = next((i for i in _INTENTS if i in raw), "other")
        except Exception as exc:  # noqa: BLE001 - routing must never fail the turn
            sp.set_attribute("router.error", str(exc))
            intent = "other"
        sp.set_attribute("router.intent", intent)
        set_output(sp, intent)
        return intent


def run_turn(
    question: str,
    *,
    settings: Settings,
    history: list[dict[str, Any]] | None = None,
    prompt_version: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    model: str | None = None,
) -> TurnResult:
    """Run one user turn end to end, emitting a full trace.

    `model` overrides the answer model only; the intent router keeps its own
    cheap model either way. It exists so poc/08 can hold the prompt fixed and
    vary the model instead -- the same experiment machinery, a different
    independent variable.
    """
    version = prompt_version or settings.prompt_version
    answer_model = model or AGENT_MODEL
    session_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"
    client = _client(settings)
    system_prompt = load_prompt(version, settings=settings)
    tools = tools_for(version)

    messages: list[dict[str, Any]] = list(history or [])
    messages.append({"role": "user", "content": question})

    ctx = ToolContext()
    result = TurnResult(question=question, answer="", session_id=session_id)
    started = time.perf_counter()

    metadata = {
        "prompt_version": version,
        "agent_model": answer_model,
        "turn_index": len(history or []) // 2,
        **(extra_metadata or {}),
    }

    # session_id / user_id propagate onto every span in the block, which is what
    # populates the Sessions view in Arize.
    with using_attributes(session_id=session_id, user_id=user_id or "anonymous"):
        with span(
            "copilot.turn",
            OpenInferenceSpanKindValues.AGENT,
            input_value=question,
            metadata=metadata,
            tags=tags,
        ) as agent_span:
            result.intent = _classify(client, question)
            agent_span.set_attribute("copilot.intent", result.intent)

            answer_parts: list[str] = []
            try:
                for _ in range(MAX_TOOL_ITERATIONS):
                    # max_tokens covers the CoT as well as the answer, so it is
                    # sized for thinking rather than for a ~150-word reply.
                    resp = client.chat.completions.create(
                        model=answer_model,
                        max_tokens=8192,
                        messages=[{"role": "system", "content": system_prompt}, *messages],
                        tools=tools,
                        extra_body=THINKING_ON,
                    )
                    if resp.usage:
                        result.input_tokens += resp.usage.prompt_tokens or 0
                        result.output_tokens += resp.usage.completion_tokens or 0

                    message = resp.choices[0].message
                    if message.content:
                        answer_parts.append(message.content)

                    if not message.tool_calls:
                        break

                    assistant_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in message.tool_calls
                        ],
                    }
                    # Inside a tool loop DeepSeek requires the CoT back in the
                    # context on every following request -- dropping it is a
                    # 400. (Outside a tool loop the opposite holds: it is not
                    # concatenated and can be omitted.) It rides on the message
                    # as an extra field, so read it defensively.
                    reasoning = getattr(message, "reasoning_content", None)
                    if reasoning:
                        assistant_message["reasoning_content"] = reasoning
                    messages.append(assistant_message)
                    for call in message.tool_calls:
                        try:
                            arguments = json.loads(call.function.arguments or "{}")
                        except json.JSONDecodeError:
                            arguments = {}
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": execute(ctx, call.function.name, arguments),
                            }
                        )
                else:
                    result.error = "max_tool_iterations_exceeded"
            except Exception as exc:  # noqa: BLE001 - one bad turn shouldn't kill a run
                result.error = f"{type(exc).__name__}: {exc}"
                agent_span.set_attribute("copilot.error", result.error)

            result.answer = "\n".join(p for p in answer_parts if p.strip()).strip()
            result.tool_calls = ctx.tool_names
            result.retrieved_doc_ids = ctx.retrieved_doc_ids
            result.retrieved_context = ctx.retrieved_context
            result.escalated = ctx.escalated
            result.latency_ms = (time.perf_counter() - started) * 1000

            agent_span.set_attribute("copilot.tool_calls", result.tool_calls)
            agent_span.set_attribute("copilot.tool_call_count", len(result.tool_calls))
            agent_span.set_attribute("copilot.escalated", result.escalated)
            agent_span.set_attribute("copilot.retrieved_doc_ids", result.retrieved_doc_ids)
            agent_span.set_attribute("copilot.prompt_version", version)
            set_output(agent_span, result.answer)

    return result


def run_conversation(
    questions: list[str],
    *,
    settings: Settings,
    prompt_version: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[TurnResult]:
    """Run several turns under one session id, carrying history between them."""
    session_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"
    history: list[dict[str, Any]] = []
    results: list[TurnResult] = []

    for question in questions:
        turn = run_turn(
            question,
            settings=settings,
            history=history,
            prompt_version=prompt_version,
            session_id=session_id,
            user_id=user_id,
            extra_metadata=extra_metadata,
        )
        results.append(turn)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": turn.answer or "(no answer)"})

    return results


def answer_only(question: str, *, settings: Settings, prompt_version: str) -> str:
    """Thin wrapper used as the experiment task in poc/08."""
    return run_turn(
        question,
        settings=settings,
        prompt_version=prompt_version,
        tags=["experiment"],
    ).answer


def load_questions() -> list[dict[str, Any]]:
    from .config import DATA_DIR

    path = DATA_DIR / "questions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

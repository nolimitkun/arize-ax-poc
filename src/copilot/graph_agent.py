"""The LangGraph/LangChain build of the Nimbus copilot.

Same logic as `agent.py`, different engine. One turn runs this graph:

    START -> classify -> agent --(tool calls)--> tools -> agent   (loop)
                           \\--(no tool calls)--> END

Everything that makes the POC's seeded failures fail is imported, not
reimplemented: the v1/v2 system prompts (`prompts.py`), the v1/v2 tool schemas
(`tools.py` -- `bind_tools` accepts the OpenAI-format dicts verbatim, so v1's
deliberately-vague descriptions survive byte for byte), the corpus, and the
tool implementations themselves via `tools.execute`, which also keeps the
per-turn trajectory record that poc/03-04 grade.

What changes is who emits the spans. The SDK engine hand-builds every span;
here `LangChainInstrumentor` traces the graph -- nodes as CHAIN spans, model
calls as LLM spans with token counts -- nested under the same manually-opened
`copilot.turn` AGENT root, so every tour script's filter keeps working. The
TOOL spans still come from `tools.py`'s own wrappers.

The graph is compiled once. Everything per-run (settings, prompt version,
answer model, the ToolContext) travels in `config["configurable"]`, which is
also what lets poc/00 compile and walk the graph offline with no credentials.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openinference.instrumentation import using_attributes
from openinference.semconv.trace import OpenInferenceSpanKindValues

from .agent import _INTENTS, _ROUTER_PROMPT, MAX_TOOL_ITERATIONS, TurnResult
from .config import AGENT_MODEL, ROUTER_MODEL, THINKING_OFF, THINKING_ON, Settings
from .prompts import load_prompt
from .tools import ToolContext, execute, tools_for
from .tracing import set_output, span


class _ChatDeepSeek(ChatDeepSeek):
    """ChatDeepSeek that echoes the CoT back inside tool loops.

    DeepSeek returns 400 if an assistant message that made tool calls comes
    back without its `reasoning_content`. ChatDeepSeek *captures* the field
    into `additional_kwargs` on the way in, but langchain-openai's message
    serializer drops unknown kwargs on the way out -- so it has to be copied
    onto the wire format here. Only tool-calling assistant messages get it,
    matching what the SDK engine does (outside a tool loop it may be omitted).
    """

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        lc_messages = self._convert_input(input_).to_messages()
        for lc_message, wire in zip(lc_messages, payload["messages"]):
            if (
                wire.get("role") == "assistant"
                and wire.get("tool_calls")
                and isinstance(lc_message, AIMessage)
                and lc_message.additional_kwargs.get("reasoning_content")
            ):
                wire["reasoning_content"] = lc_message.additional_kwargs["reasoning_content"]
        return payload


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    # Reset by `classify` at the start of every turn, so a persistent
    # (checkpointed) thread doesn't carry one turn's loop count into the next.
    agent_calls: int
    error: str | None


def _conf(config: RunnableConfig, key: str) -> Any:
    return config["configurable"][key]


def _settings(config: RunnableConfig) -> Settings:
    return _conf(config, "settings")


def classify(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Intent routing node -- flash model, thinking off.

    Thinking off is load-bearing, same as the SDK engine: a 32-token budget
    would otherwise be spent entirely on the CoT, leaving nothing to parse.
    """
    settings = _settings(config)
    question = str(state["messages"][-1].content)
    try:
        llm = _ChatDeepSeek(
            model=ROUTER_MODEL,
            api_key=settings.deepseek_api_key,
            api_base=settings.deepseek_base_url,
            max_tokens=32,
            extra_body=THINKING_OFF,
        )
        raw = str(
            llm.invoke(
                [SystemMessage(_ROUTER_PROMPT), {"role": "user", "content": question}]
            ).content
        ).strip().lower()
        intent = next((i for i in _INTENTS if i in raw), "other")
    except Exception:  # noqa: BLE001 - routing must never fail the turn
        intent = "other"
    return {"intent": intent, "agent_calls": 0, "error": None}


def agent(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """The answer model with the version's tools bound.

    The system prompt arrives pre-resolved in the config rather than being
    looked up here: this node runs once per tool round, and `load_prompt("hub")`
    is a remote fetch each time -- mid-turn the label could move, or a
    transient failure could swap in the local fallback, leaving one turn
    answered by two different prompts. `run_turn` resolves it once, like the
    SDK engine does.
    """
    settings = _settings(config)
    version = _conf(config, "prompt_version")
    llm = _ChatDeepSeek(
        model=_conf(config, "answer_model"),
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_base_url,
        # Covers the CoT as well as the answer, so it is sized for thinking
        # rather than for a ~150-word reply.
        max_tokens=8192,
        extra_body=THINKING_ON,
    ).bind_tools(tools_for(version))
    system = SystemMessage(_conf(config, "system_prompt"))
    response = llm.invoke([system, *state["messages"]])
    return {"messages": [response], "agent_calls": state["agent_calls"] + 1}


def _all_tool_calls(message: Any) -> list[dict[str, Any]]:
    """Valid and malformed tool calls together.

    LangChain routes a tool call whose arguments fail to parse as JSON into
    `invalid_tool_calls` and leaves `tool_calls` empty. Reading only the latter
    silently ends the turn with whatever partial text exists and no error --
    the SDK engine instead answers the call with an invalid-arguments result
    and lets the model retry, so both lists have to flow into the loop.
    """
    return [
        *(getattr(message, "tool_calls", None) or []),
        *(getattr(message, "invalid_tool_calls", None) or []),
    ]


def run_tools(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Dispatch tool calls through `tools.execute`.

    Not the prebuilt ToolNode: execution has to flow through the existing
    implementations so the ToolContext trajectory record, the retrieved-context
    accumulation, and the escalation flag stay identical to the SDK engine --
    and so the TOOL spans keep their names and attributes.
    """
    ctx: ToolContext = _conf(config, "tool_ctx")
    last = state["messages"][-1]
    results = [
        ToolMessage(
            content=execute(ctx, call["name"], call["args"] or {}),
            tool_call_id=call["id"],
        )
        for call in getattr(last, "tool_calls", None) or []
    ]
    for call in getattr(last, "invalid_tool_calls", None) or []:
        # Mirrors the SDK engine's unparseable-arguments path: dispatch with
        # empty args, which records the attempt as failed on the trajectory and
        # returns an invalid-arguments message the model can react to.
        results.append(
            ToolMessage(
                content=execute(ctx, call.get("name") or "", {}),
                tool_call_id=call.get("id") or "invalid-tool-call",
            )
        )
    return {"messages": results}


def decide_next(state: GraphState) -> str:
    """Loop into tools, stop cleanly, or stop because the cap is hit.

    The iteration cap is enforced here rather than via recursion_limit --
    LangGraph counts super-steps, not tool rounds, and an in-graph counter is
    exact and testable offline. recursion_limit stays as a backstop only.
    """
    last = state["messages"][-1]
    if not _all_tool_calls(last):
        return "end"
    if state["agent_calls"] >= MAX_TOOL_ITERATIONS:
        return "give_up"
    return "tools"


def give_up(state: GraphState) -> dict[str, Any]:
    """Cap hit with tool calls still pending.

    The pending calls get stub ToolMessages as well as the error flag: on a
    checkpointed thread the transcript persists into the next turn, and an
    assistant message whose tool calls were never answered would 400 the next
    request. The SDK engine can't hit this -- it rebuilds history from plain
    text -- so the stubs are what keeps the two engines' behaviour aligned.
    """
    last = state["messages"][-1]
    stubs = [
        ToolMessage(
            content="Tool loop aborted: iteration cap reached.",
            tool_call_id=call.get("id") or "invalid-tool-call",
        )
        for call in _all_tool_calls(last)
    ]
    return {"error": "max_tool_iterations_exceeded", "messages": stubs}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify", classify)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("give_up", give_up)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "agent")
    graph.add_conditional_edges(
        "agent", decide_next, {"tools": "tools", "give_up": "give_up", "end": END}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("give_up", END)
    return graph


_CHECKPOINTER = MemorySaver()
_GRAPH = build_graph().compile(checkpointer=_CHECKPOINTER)

# Backstop only -- the real cap is `decide_next`. Per turn: classify +
# up to MAX_TOOL_ITERATIONS agent steps + the tools steps between them + give_up.
RECURSION_LIMIT = 2 * MAX_TOOL_ITERATIONS + 3


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
    _thread_id: str | None = None,
    _turn_index: int | None = None,
) -> TurnResult:
    """Run one user turn through the graph. Same contract as `agent.run_turn`.

    `_thread_id` is how `run_conversation` carries history: when set, the
    checkpointer already holds the earlier turns and only the new question is
    sent. Without it, each call is a fresh thread seeded from `history` --
    which is what the tour scripts that pass explicit history rely on.
    """
    version = prompt_version or settings.prompt_version
    answer_model = model or AGENT_MODEL
    session_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"

    ctx = ToolContext()
    result = TurnResult(question=question, answer="", session_id=session_id)
    started = time.perf_counter()

    if _thread_id:
        inputs: list[Any] = [{"role": "user", "content": question}]
        thread_id = _thread_id
    else:
        inputs = [*(history or []), {"role": "user", "content": question}]
        thread_id = f"turn-{uuid.uuid4().hex[:12]}"

    metadata = {
        "prompt_version": version,
        "agent_model": answer_model,
        "turn_index": _turn_index if _turn_index is not None else len(history or []) // 2,
        "engine": "langgraph",
        # session_id doubles as metadata: Arize Sessions read the session.id
        # attribute, LangSmith Threads read metadata -- see agent.py.
        "session_id": session_id,
        "user_id": user_id or "anonymous",
        **(extra_metadata or {}),
    }

    config: RunnableConfig = {
        "configurable": {
            "settings": settings,
            "tool_ctx": ctx,
            "prompt_version": version,
            # Resolved once per turn -- "hub" is a remote fetch, and the agent
            # node runs once per tool round (see the node's docstring).
            "system_prompt": load_prompt(version, settings=settings),
            "answer_model": answer_model,
            "thread_id": thread_id,
        },
        "recursion_limit": RECURSION_LIMIT,
    }

    # session_id / user_id propagate onto every span in the block -- including
    # the ones LangChainInstrumentor creates -- which populates Sessions.
    with using_attributes(session_id=session_id, user_id=user_id or "anonymous"):
        with span(
            "copilot.turn",
            OpenInferenceSpanKindValues.AGENT,
            input_value=question,
            metadata=metadata,
            tags=tags,
        ) as agent_span:
            final: dict[str, Any] = {}
            try:
                final = _GRAPH.invoke({"messages": inputs}, config)
                result.error = final.get("error")
            except GraphRecursionError:
                result.error = "max_tool_iterations_exceeded"
                final = _GRAPH.get_state(config).values
            except Exception as exc:  # noqa: BLE001 - one bad turn shouldn't kill a run
                result.error = f"{type(exc).__name__}: {exc}"
                agent_span.set_attribute("copilot.error", result.error)

            result.intent = final.get("intent", "unknown")
            agent_span.set_attribute("copilot.intent", result.intent)

            # Everything after the question we just appended is this turn's.
            messages = final.get("messages", [])
            last_human = max(
                (i for i, m in enumerate(messages) if m.type == "human"), default=-1
            )
            answer_parts: list[str] = []
            for message in messages[last_human + 1 :]:
                if isinstance(message, AIMessage):
                    if isinstance(message.content, str) and message.content.strip():
                        answer_parts.append(message.content)
                    usage = message.usage_metadata or {}
                    result.input_tokens += usage.get("input_tokens", 0)
                    result.output_tokens += usage.get("output_tokens", 0)

            result.answer = "\n".join(answer_parts).strip()
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
    """Several turns under one session id.

    History rides in the checkpointer under `thread_id=session_id` -- the
    LangGraph-native shape of what `agent.run_conversation` does by hand.
    """
    session_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"
    return [
        run_turn(
            question,
            settings=settings,
            prompt_version=prompt_version,
            session_id=session_id,
            user_id=user_id,
            extra_metadata=extra_metadata,
            _thread_id=session_id,
            _turn_index=index,
        )
        for index, question in enumerate(questions)
    ]

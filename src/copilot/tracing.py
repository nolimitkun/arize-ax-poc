"""OpenTelemetry / OpenInference wiring for the copilot.

`register()` must run before the first LLM call, or the auto-instrumentor has
nothing to patch. Every entry point therefore calls `init_tracing()` first.

The model is DeepSeek, reached over its OpenAI-compatible endpoint, so the
OpenAI instrumentor is the right one -- it patches `openai.OpenAI`, which is the
client we actually call, regardless of the base URL behind it.

Where the spans go is a separate, fan-out decision (COPILOT_OBSERVABILITY):
one tracer provider, one span processor per enabled backend -- Arize via
arize.otel.register(), LangSmith via its OTLP/HTTP ingestion endpoint.

Docs: https://arize.com/docs/ax/get-started/get-started-tracing
      https://arize.com/docs/ax/integrations/llm-providers/openai/openai-tracing
      https://docs.smith.langchain.com/observability/how_to_guides/trace_with_opentelemetry
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace as trace_api
from opentelemetry.trace import Span, Status, StatusCode

from .config import Settings

_TRACER: trace_api.Tracer | None = None
_INITIALIZED = False


def _langsmith_processor(settings: Settings, project: str):
    """A second OTLP lane into LangSmith.

    LangSmith ingests plain OTLP/HTTP and maps OpenInference attributes, so
    the exact spans Arize receives -- manual AGENT/TOOL spans and the
    auto-instrumented LLM spans -- fan out there unchanged. The project is
    named by header, not by resource attribute.
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    exporter = OTLPSpanExporter(
        endpoint=settings.langsmith_otlp_endpoint,
        headers={
            "x-api-key": settings.langsmith_api_key,
            "Langsmith-Project": project,
        },
    )
    return BatchSpanProcessor(exporter)


def init_tracing(settings: Settings, project_name: str | None = None) -> trace_api.Tracer:
    """Register the enabled exporters and instrument one SDK. Idempotent.

    COPILOT_OBSERVABILITY picks the backend(s): `arize` keeps today's
    register() path, `both` adds a LangSmith span processor to the same
    provider (same spans, two backends, tokens counted once), `langsmith`
    builds a plain SDK provider with no Arize wiring at all.
    """
    global _TRACER, _INITIALIZED
    if _INITIALIZED and _TRACER is not None:
        return _TRACER

    project = project_name or settings.arize_project_name

    if settings.arize_enabled:
        # Imported lazily so that importing this module doesn't pull in the
        # whole OTel stack for scripts that only need the platform client.
        from arize.otel import register

        # endpoint is passed explicitly rather than left to register()'s
        # default, so the region resolved in config.py actually applies.
        # The LangSmith lane must ride in through `span_processors=`: Arize's
        # TracerProvider marks its own exporter as a "default" processor, and
        # a later add_span_processor() call *shuts it down and removes it* --
        # `both` mode would silently become langsmith-only. Passing custom
        # processors up front clears that default flag instead.
        extra = [_langsmith_processor(settings, project)] if settings.langsmith_enabled else []
        tracer_provider = register(
            space_id=settings.arize_space_id,
            api_key=settings.arize_api_key,
            project_name=project,
            endpoint=settings.arize_otlp_endpoint,
            span_processors=extra or None,
        )
    else:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        tracer_provider = TracerProvider(resource=Resource.create({"model_id": project}))
        trace_api.set_tracer_provider(tracer_provider)
        tracer_provider.add_span_processor(_langsmith_processor(settings, project))

    # One instrumentor, never both. The LangGraph engine's LLM calls go through
    # langchain-openai, which drives the `openai` SDK underneath -- with both
    # instrumentors active every call would emit two LLM spans and double-count
    # tokens and cost in AX.
    if settings.impl == "langgraph":
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    else:
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    _TRACER = tracer_provider.get_tracer(__name__)
    _INITIALIZED = True
    return _TRACER


def tracer() -> trace_api.Tracer:
    if _TRACER is None:
        raise RuntimeError("init_tracing() must be called before tracer()")
    return _TRACER


def flush() -> None:
    """Force-export buffered spans.

    The exporter batches, so a short-lived CLI script can exit before its spans
    leave the process. Every poc/ script calls this before returning.
    """
    provider = trace_api.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()


def _set_io(span: Span, value: Any, attr: str) -> None:
    """Set input.value / output.value, JSON-encoding non-strings."""
    if value is None:
        return
    if isinstance(value, str):
        span.set_attribute(attr, value)
        span.set_attribute(attr.replace(".value", ".mime_type"), "text/plain")
    else:
        span.set_attribute(attr, json.dumps(value, default=str))
        span.set_attribute(attr.replace(".value", ".mime_type"), "application/json")


@contextmanager
def span(
    name: str,
    kind: OpenInferenceSpanKindValues,
    *,
    input_value: Any = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[Span]:
    """Open a manually-instrumented span of a given OpenInference kind.

    Auto-instrumentation only covers the LLM calls. The AGENT / CHAIN /
    RETRIEVER / TOOL spans that give the trace tree its shape -- and that AX's
    agent-trajectory views key off -- have to be created explicitly.
    """
    with tracer().start_as_current_span(name) as sp:
        sp.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind.value)
        _set_io(sp, input_value, SpanAttributes.INPUT_VALUE)
        if metadata:
            sp.set_attribute(SpanAttributes.METADATA, json.dumps(metadata, default=str))
        if tags:
            sp.set_attribute(SpanAttributes.TAG_TAGS, tags)
        try:
            yield sp
        except Exception as exc:
            sp.set_status(Status(StatusCode.ERROR, str(exc)))
            sp.record_exception(exc)
            raise
        else:
            sp.set_status(Status(StatusCode.OK))


def set_output(sp: Span, value: Any) -> None:
    _set_io(sp, value, SpanAttributes.OUTPUT_VALUE)

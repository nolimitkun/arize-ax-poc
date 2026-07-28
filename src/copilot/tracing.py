"""OpenTelemetry / OpenInference wiring for the copilot.

`register()` must run before the first LLM call, or the auto-instrumentor has
nothing to patch. Every entry point therefore calls `init_tracing()` first.

The model is DeepSeek, reached over its OpenAI-compatible endpoint, so the
OpenAI instrumentor is the right one -- it patches `openai.OpenAI`, which is the
client we actually call, regardless of the base URL behind it.

Docs: https://arize.com/docs/ax/get-started/get-started-tracing
      https://arize.com/docs/ax/integrations/llm-providers/openai/openai-tracing
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


def init_tracing(settings: Settings, project_name: str | None = None) -> trace_api.Tracer:
    """Register the Arize exporter and instrument the OpenAI SDK. Idempotent."""
    global _TRACER, _INITIALIZED
    if _INITIALIZED and _TRACER is not None:
        return _TRACER

    # Imported lazily so that importing this module doesn't pull in the whole
    # OTel stack for scripts that only need the platform client.
    from arize.otel import register
    from openinference.instrumentation.openai import OpenAIInstrumentor

    tracer_provider = register(
        space_id=settings.arize_space_id,
        api_key=settings.arize_api_key,
        project_name=project_name or settings.arize_project_name,
    )
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

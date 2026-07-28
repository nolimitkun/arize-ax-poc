#!/usr/bin/env python
"""Step 02 -- Instrument: customise traces, and redact PII.

Three things AX keys off that auto-instrumentation alone won't give you:

  1. custom span attributes  -> filterable/groupable in the trace view
  2. metadata + tags         -> the dimensions you slice dashboards by
  3. masking                 -> keeping PII out of the platform entirely

Docs: https://arize.com/docs/ax/instrument/customize-your-traces
      https://arize.com/docs/ax/instrument/mask-and-redact-data
"""

from __future__ import annotations

import re

import typer

from _common import console, done, header, look_at, table

app = typer.Typer(add_completion=False)

# Turns that carry PII, used to demonstrate both redaction strategies.
PII_TURNS = [
    "My email is ana.torres@acme.example and my card ending 4242 was charged twice "
    "on order NMB-4471 - what happened?",
    "Contact me at +1 415 555 0132. Why did order NMB-4489 fail?",
]

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
CARD = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def scrub(text: str) -> str:
    """Redact at the application boundary, before the span is ever created.

    This is the strategy to prefer when you know the shape of your PII: the
    sensitive value never enters the SDK, so there is nothing to leak even if
    tracing is misconfigured downstream.
    """
    text = EMAIL.sub("[EMAIL_REDACTED]", text)
    text = CARD.sub("[CARD_REDACTED]", text)
    return PHONE.sub("[PHONE_REDACTED]", text)


@app.command()
def main(
    prompt_version: str = typer.Option("", help="Override COPILOT_PROMPT_VERSION"),
) -> None:
    settings = header(
        "02",
        "Instrument: custom attributes, metadata, tags, masking",
        "customize-your-traces · mask-and-redact-data",
    )

    from openinference.semconv.trace import OpenInferenceSpanKindValues

    from copilot.agent import run_turn
    from copilot.tracing import flush, init_tracing, set_output, span

    version = prompt_version or settings.prompt_version
    init_tracing(settings)

    # ---- 1. Application-level redaction -------------------------------
    console.print("[bold]1. Application-level redaction[/bold] (scrub before tracing)\n")
    for raw in PII_TURNS:
        cleaned = scrub(raw)
        console.print(f"  [red]raw    [/red] {raw[:78]}")
        console.print(f"  [green]scrubbed[/green] {cleaned[:78]}\n")
        run_turn(
            cleaned,
            settings=settings,
            prompt_version=version,
            user_id="u_pii_demo",
            tags=["pii-redacted", "demo"],
            extra_metadata={
                "redaction": "application_level",
                "pii_types_removed": "email,card,phone",
            },
        )

    # ---- 2. SDK-level masking -----------------------------------------
    # TraceConfig can suppress whole attribute families. Useful as a blanket
    # safety net when you cannot enumerate your PII patterns -- the tradeoff is
    # that you lose the input/output text in the UI, so evals that read
    # input.value stop working on those spans.
    console.print("[bold]2. SDK-level masking[/bold] (TraceConfig / env vars)\n")
    table(
        "TraceConfig options",
        ["setting", "env var", "effect"],
        [
            ["hide_inputs", "OPENINFERENCE_HIDE_INPUTS", "drops input.value"],
            ["hide_outputs", "OPENINFERENCE_HIDE_OUTPUTS", "drops output.value"],
            ["hide_input_messages", "OPENINFERENCE_HIDE_INPUT_MESSAGES", "drops prompt messages"],
            ["hide_input_text", "OPENINFERENCE_HIDE_INPUT_TEXT", "drops message text only"],
            ["hide_llm_invocation_parameters", "OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS", "drops model params"],
        ],
    )
    console.print(
        "\n[dim]Applied at instrument() time:\n"
        "  from openinference.instrumentation import TraceConfig\n"
        "  OpenAIInstrumentor().instrument(\n"
        "      tracer_provider=tp, config=TraceConfig(hide_input_text=True))\n"
        "This POC leaves masking off so later eval steps can read span text.[/dim]\n"
    )

    # ---- 3. Custom attributes on a hand-rolled span --------------------
    console.print("[bold]3. Custom attributes, metadata and tags[/bold]\n")
    with span(
        "billing.audit_check",
        OpenInferenceSpanKindValues.CHAIN,
        input_value={"order_id": "NMB-4471", "check": "duplicate_charge"},
        metadata={
            "team": "billing-ops",
            "runbook": "RB-114",
            "customer_tier": "business",
            "region": "us-east",
        },
        tags=["audit", "billing", "synthetic"],
    ) as sp:
        # Arbitrary attributes become filterable columns in the trace view.
        sp.set_attribute("billing.duplicate_found", False)
        sp.set_attribute("billing.charges_inspected", 3)
        sp.set_attribute("billing.window_days", 30)
        set_output(sp, {"duplicate_found": False, "charges_inspected": 3})
    console.print("  emitted [bold]billing.audit_check[/bold] with custom attrs + tags\n")

    flush()

    look_at(
        "Traces → filter on `tag.tags` — the `pii-redacted` and `audit` turns.",
        "Open a redacted turn: input.value shows [EMAIL_REDACTED], not the address.",
        "Open billing.audit_check → Attributes: billing.* fields and the metadata block.",
        "Try a filter like `attributes.metadata['team'] = 'billing-ops'`.",
    )
    done("poc/03_query_spans.py — export spans and find the failures")


if __name__ == "__main__":
    app()

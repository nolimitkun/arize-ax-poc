#!/usr/bin/env python
"""Step 00 -- Self-check. Runs offline; no API keys, no network.

Verifies everything that can be verified without credentials, so that when you
do run the tour a failure means "credentials/platform", not "the repo is
broken". Checks:

  * the KB loads, and the deliberate refund gap is really a gap
  * the question fixture is well-formed
  * the code evaluators fire on the failure modes they're supposed to catch
  * v1 and v2 prompts differ in the ways the POC claims
  * eval / annotation dataframes match the Arize SDK's column patterns
"""

from __future__ import annotations

import sys

import pandas as pd

from _common import console, table

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        console.print(f"  [green]✓[/green] {name}")
    else:
        console.print(f"  [red]✗[/red] {name} [dim]{detail}[/dim]")
        FAILURES.append(name)


def check_knowledge_base() -> None:
    console.print("\n[bold]Knowledge base[/bold]")
    from copilot.kb import _index, search

    chunks, _, _, _ = _index()
    docs = {c.source for c in chunks}
    check(f"corpus loads ({len(chunks)} chunks, {len(docs)} docs)", len(chunks) > 40)

    # The gap is the whole point. Note what is NOT asserted: that retrieval
    # returns nothing. It returns weak, adjacent matches (a chunk mentioning
    # "key rotation policy" scores on the word "policy"), and that is desirable
    # -- plausible-but-irrelevant context is exactly what tempts a model into
    # inventing an answer. What must hold is that nothing retrieved actually
    # answers the question.
    corpus_text = " ".join(c.text.lower() for c in chunks)
    check(
        "no document mentions refunds at all",
        "refund" not in corpus_text,
        "a doc mentions refunds -- the hallucination test is compromised",
    )

    for query in ("What is your refund policy?", "Do I get a prorated refund if I cancel?"):
        hits = search(query)
        answering = [c.doc_id for c, _ in hits if "refund" in c.text.lower()]
        check(
            f"no chunk answers {query[:40]!r}",
            not answering,
            f"these would answer it: {answering}",
        )

    # ...but ordinary questions must retrieve well, or the agent fails for the
    # wrong reason and the POC measures nothing.
    for query, expect in [
        ("What scopes can an API key have?", "api-keys.md"),
        ("How many times do you retry a failed webhook?", "webhooks.md"),
        ("What is the uptime SLA on Business?", "status-and-incidents.md"),
        ("How do I fix schema_drift?", "troubleshooting-runs.md"),
    ]:
        hits = search(query)
        top = hits[0][0].source if hits else "(nothing)"
        check(f"retrieval: {query[:44]!r} → {expect}", top == expect, f"got {top}")


def check_questions() -> None:
    console.print("\n[bold]Question fixture[/bold]")
    from copilot.agent import load_questions

    questions = load_questions()
    check(f"questions.jsonl parses ({len(questions)} rows)", len(questions) >= 40)

    required = {"id", "question", "expected_behavior", "expected_tools", "topic", "failure_mode"}
    check("every row has the required fields", all(required <= set(q) for q in questions))
    check("ids are unique", len({q["id"] for q in questions}) == len(questions))

    modes = {q["failure_mode"] for q in questions if q["failure_mode"]}
    for mode in ("hallucination", "wrong_tool", "missing_escalation", "verbosity"):
        count = sum(1 for q in questions if q["failure_mode"] == mode)
        check(f"seeded failure mode `{mode}` present ({count} questions)", count >= 4)

    seeded = sum(1 for q in questions if q["failure_mode"])
    share = seeded / len(questions)
    check(
        f"seeded failures are {share:.0%} of the set (want 30-70%)",
        0.3 <= share <= 0.7,
    )


def check_code_evaluators() -> None:
    console.print("\n[bold]Code evaluators[/bold]")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import importlib

    offline = importlib.import_module("04_offline_evals")
    observe = importlib.import_module("03_query_spans")

    hallucinated = observe.check_ungrounded(
        "What is your refund policy?",
        "You have 30 days from the charge date to request a full refund.",
        [],
    )
    check("ungrounded check catches an invented refund window", hallucinated)

    hedged = observe.check_ungrounded(
        "What is your refund policy?",
        "Our documentation doesn't cover refunds, so I don't want to guess. "
        "I can escalate this to a human.",
        [],
    )
    check("ungrounded check accepts an honest refusal", not hedged)

    unrelated = observe.check_ungrounded(
        "What scopes can an API key have?", "read, write, run and admin.", ["api-keys.md#Scopes"]
    )
    check("ungrounded check ignores non-refund questions", not unrelated)

    check(
        "wrong-tool check fires when lookup_order is skipped",
        observe.check_wrong_tool(["lookup_order"], ["search_docs"]),
    )
    check(
        "wrong-tool check passes when lookup_order is used",
        not observe.check_wrong_tool(["lookup_order"], ["lookup_order"]),
    )
    check(
        "escalation check fires on a missed escalation",
        observe.check_missing_escalation(["escalate_ticket"], ["search_docs"]),
    )
    check("verbosity check fires past 250 words", observe.check_verbose("word " * 300))
    check("verbosity check passes a short answer", not observe.check_verbose("short answer"))

    # The same logic, as used against the eval dataframe in step 04.
    row = pd.Series({"expected_tools": "lookup_order", "tool_calls": "search_docs"})
    label, score, _ = offline.eval_tool_selection(row)
    check("eval_tool_selection labels a miss `incorrect`", label == "incorrect" and score == 0.0)

    row = pd.Series({"expected_tools": "escalate_ticket", "tool_calls": ""})
    label, score, _ = offline.eval_escalation(row)
    check("eval_escalation labels a miss `missed`", label == "missed" and score == 0.0)

    label, score, _ = offline.eval_conciseness(pd.Series({"answer_words": 400}))
    check("eval_conciseness labels 400 words `verbose`", label == "verbose" and score == 0.0)

    # Regression: both Phoenix columns start with the evaluator name, and
    # selecting by prefix silently produced all-NaN scores that Arize rejected.
    graded = pd.DataFrame(
        {
            "groundedness_execution_details": [
                {"status": "COMPLETED", "exceptions": [], "execution_seconds": 1.0}
            ],
            "groundedness_score": [
                {"name": "groundedness", "score": 0.0, "label": "hallucinated",
                 "explanation": "invented a refund window"}
            ],
        }
    )
    parsed = offline.parse_judge_output(graded, "groundedness")
    check("judge output parses past the _execution_details column", parsed is not None)
    if parsed:
        labels, scores, expl = parsed
        check(
            "judge score/label read from the right column",
            scores.iloc[0] == 0.0 and labels.iloc[0] == "hallucinated" and expl.iloc[0],
            f"got score={scores.iloc[0]!r} label={labels.iloc[0]!r}",
        )
    check(
        "missing judge column is reported, not silently NaN",
        offline.parse_judge_output(pd.DataFrame({"other": [1]}), "groundedness") is None,
    )
    check(
        "per-row judge exceptions are surfaced",
        offline.judge_failures(
            pd.DataFrame({"groundedness_execution_details": [{"exceptions": ["boom"]}]}),
            "groundedness",
        )
        != [],
    )

    # A failed judge row yields label=None/score=NaN. Uploading it alongside
    # good rows makes Arize reject the whole batch, so it must be filterable.
    mixed = pd.DataFrame(
        {
            "groundedness_score": [
                {"score": 1.0, "label": "grounded", "explanation": "ok"},
                {},
            ]
        }
    )
    labels, scores, _ = offline.parse_judge_output(mixed, "groundedness")
    usable = labels.notna() & scores.notna()
    check(
        "failed judge rows are distinguishable from graded ones",
        usable.tolist() == [True, False],
        f"got {usable.tolist()}",
    )


def check_experiment_statistics() -> None:
    """The verdict in poc/08 rests on these, so they get pinned down."""
    console.print("\n[bold]Experiment significance[/bold]")
    from importlib import import_module

    exp = import_module("08_experiments")

    check("no disagreement -> p=1", exp.mcnemar_p(0, 0) == 1.0)
    check(
        "1 vs 0 is not significant (the noise case that mis-declared a win)",
        exp.mcnemar_p(0, 1) > 0.05,
        f"p={exp.mcnemar_p(0, 1):.3f}",
    )
    check(
        "6 vs 0 is significant",
        exp.mcnemar_p(0, 6) < 0.05,
        f"p={exp.mcnemar_p(0, 6):.3f}",
    )
    check("symmetric in its arguments", exp.mcnemar_p(0, 6) == exp.mcnemar_p(6, 0))
    check(
        "an even split is never significant",
        exp.mcnemar_p(5, 5) == 1.0,
        f"p={exp.mcnemar_p(5, 5):.3f}",
    )
    # Exact two-sided binomial: 2 * P(X<=0) = 2 * (1/2)^6 = 0.03125
    check(
        "matches the exact binomial value",
        abs(exp.mcnemar_p(0, 6) - 0.03125) < 1e-9,
        f"got {exp.mcnemar_p(0, 6)}",
    )

    # Pairing: v2 fixes 4 rows, breaks 0 -> significant improvement.
    base = pd.DataFrame({"example_id": list(range(6)), "eval.groundedness.score": [0, 0, 0, 0, 1, 1]})
    cand = pd.DataFrame({"example_id": list(range(6)), "eval.groundedness.score": [1, 1, 1, 1, 1, 1]})
    paired = exp.paired_verdict(base, cand, "groundedness")
    check("paired verdict finds the fixed rows", paired is not None and paired[:2] == (0, 4), str(paired))
    check("unpairable frames return None", exp.paired_verdict(
        pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1]}), "groundedness") is None)
    # An unpaired evaluator must not be called significant on the strength of a
    # big delta -- magnitude alone says nothing about how many rows moved.
    from pathlib import Path

    source = (Path(__file__).with_name("08_experiments.py")).read_text(encoding="utf-8")
    unpaired_branch = source.split("if paired is None:")[1].split("else:")[0]
    check(
        "unpaired results are never declared significant",
        "significant = False" in unpaired_branch
        and "abs(delta)" not in unpaired_branch,
        unpaired_branch.strip()[:120],
    )


def check_judge_verdict_parsing() -> None:
    """`ungrounded` must not read as `grounded` -- it inverts the verdict."""
    console.print("\n[bold]Judge verdict parsing[/bold]")
    import math

    from copilot.evals import parse_verdict

    for text, expected in [
        ("grounded\nEvery claim is in the docs.", "grounded"),
        ("hallucinated\nInvented a 30-day window.", "hallucinated"),
        ("Verdict: grounded", "grounded"),
        ("**hallucinated**", "hallucinated"),
        # Markdown emphasis with underscores. A `[a-z_]+` character class makes
        # this one unknown token and scores a caught hallucination as an error.
        ("__hallucinated__", "hallucinated"),
        ("GROUNDED.", "grounded"),
        # The bug: substring matching scored these as passes.
        ("ungrounded\nNo support for the refund claim.", "error"),
        ("not hallucinated", "error"),
        ("I cannot determine this.", "error"),
        ("", "error"),
    ]:
        label, score, _ = parse_verdict(text)
        check(
            f"{(text.splitlines() or [''])[0][:34]!r:<38} -> {expected}",
            label == expected,
            f"got {label}",
        )
        if expected == "error":
            check("  ...and scores NaN, so the row leaves the mean", math.isnan(score))

    _, _, why = parse_verdict("maybe?")
    check("unparseable output is quoted back for debugging", "maybe?" in why, why)


def check_prompt_hub_plumbing() -> None:
    """Every failure here is silent: it serves a local copy and calls it success."""
    console.print("\n[bold]Prompt Hub plumbing[/bold]")
    from importlib import import_module

    from copilot.prompts import system_text

    class Msg:
        def __init__(self, role, content):
            self.role, self.content = role, content

    class Version:
        def __init__(self, messages):
            self.messages = messages

    class PromptWithVersion:
        """What prompts.get() actually returns -- messages hang off .version."""

        def __init__(self, messages):
            self.version = Version(messages)

    check(
        "messages are read from .version, not off the prompt",
        system_text(PromptWithVersion([Msg("SYSTEM", "hello")])) == "hello",
    )
    check(
        "upper-case SYSTEM role is matched",
        system_text(Version([Msg("SYSTEM", "hello")])) == "hello",
    )

    class Enumish:
        value = "SYSTEM"

    check(
        "an enum role is unwrapped, not str()'d into 'MessageRole.SYSTEM'",
        system_text(Version([Msg(Enumish(), "hello")])) == "hello",
    )
    check("non-system messages are ignored", system_text(Version([Msg("USER", "hi")])) == "")

    # The dedupe helper reads the prompt name from its argument: poc/09 imports
    # PROMPT_NAME inside main(), so a module-global reference raises NameError
    # into a broad except and republishes everything on every run.
    hub = import_module("09_prompt_hub")
    import inspect

    check(
        "list_versions takes the prompt name rather than reaching for a global",
        "prompt_name" in inspect.signature(hub.list_versions).parameters,
    )
    versions = [Version([Msg("SYSTEM", "v1 text")]), Version([Msg("SYSTEM", "v2 text")])]
    for v, ident in zip(versions, ("id-1", "id-2")):
        v.id = ident
    check("find_version matches an already-published version", hub.find_version(versions, "v2 text") == "id-2")
    check("find_version ignores whitespace drift", hub.find_version(versions, "  v1 text\n") == "id-1")
    check("find_version returns '' for new text", hub.find_version(versions, "v3 text") == "")

    # Everything after the text comparison -- the probe, the hedge check, the
    # "this behaviour came from Prompt Hub" conclusion -- is a claim about the
    # version just promoted. If the hub served a different one, continuing
    # attributes that version's behaviour to this one and still reports success.
    from pathlib import Path

    source = (Path(__file__).with_name("09_prompt_hub.py")).read_text(encoding="utf-8")
    mismatch = source.split("if published.strip() != expected.strip():")[1].split(
        "matches the promoted version"
    )[0]
    check(
        "a prompt-text mismatch aborts rather than warning and continuing",
        "raise typer.Exit(1)" in mismatch,
        mismatch.strip()[:120],
    )


def check_monitor_metrics() -> None:
    """A monitor on a metric that never arrives is green forever and pages nobody."""
    console.print("\n[bold]Monitor metric resolution[/bold]")
    from importlib import import_module

    mon = import_module("10_monitors")

    available = ["eval.Groundedness.score", "eval.conciseness.score"]
    metric, note = mon.resolve_metric("eval.groundedness.score", available)
    check("a case-mismatched eval column is repointed", metric == "eval.Groundedness.score", metric)
    check("...and the repoint is reported", bool(note), note)

    metric, note = mon.resolve_metric("eval.conciseness.score", available)
    check("an exact match is used", metric == "eval.conciseness.score")
    check("...but batch-only columns are called out as going blind", "age out" in note, note)

    metric, note = mon.resolve_metric("eval.missing.score", available)
    check("a metric with no data is flagged", "cannot fire" in note, note)

    metric, note = mon.resolve_metric("latencyP95Ms", available)
    check("non-eval metrics are left alone", metric == "latencyP95Ms" and not note)

    # The real live case: step 04's batch writes eval.groundedness.score once,
    # step 05's continuous evaluator keeps writing eval.Groundedness.score.
    # Preferring the exact spelling picks the dead one, and the monitor stops
    # seeing data as those spans age out of its window.
    both = ["eval.groundedness.score", "eval.Groundedness.score"]
    live = {"eval.Groundedness.score"}
    metric, note = mon.resolve_metric("eval.groundedness.score", both, live)
    check(
        "an exact match loses to the continuously-written column",
        metric == "eval.Groundedness.score",
        metric,
    )
    check("...and the reason is given", "continuous" in note.lower(), note)

    metric, note = mon.resolve_metric("eval.Groundedness.score", both, live)
    check(
        "already pointing at the continuous column is silent",
        metric == "eval.Groundedness.score" and not note,
        note,
    )

    metric, note = mon.resolve_metric("eval.escalation_appropriate.score", both, live)
    check(
        "an eval with no continuous writer at all is still flagged",
        "cannot fire" in note,
        note,
    )

    # Verified against the live schema by introspection. createPerformanceMonitor
    # has no field for an eval column at all -- its metric is the classic-ML
    # PerformanceMetric enum -- so an eval monitor has to be a data-quality one.
    check(
        "eval monitors use the data-quality mutation, not the performance one",
        "createDataQualityMonitor" in mon.CREATE_MONITOR
        and "createPerformanceMonitor" not in mon.CREATE_MONITOR,
    )
    inputs = mon.monitor_inputs("space-1", "proj")
    required = {"name", "operator", "dimensionName", "dimensionCategory", "dataQualityMetric"}
    check(
        "every monitor carries the fields the input type requires",
        all(required <= set(m) for m in inputs),
        str([sorted(required - set(m)) for m in inputs]),
    )
    check(
        "eval dimensions are the full column name and categorised llmEval",
        all(
            m["dimensionName"].startswith("eval.") and m["dimensionCategory"] == "llmEval"
            for m in inputs
            if m["dimensionName"].startswith("eval.")
        ),
    )
    check(
        "latency is a span property named latency_ms, not a performance metric",
        any(
            m["dimensionName"] == "latency_ms" and m["dimensionCategory"] == "spanProperty"
            for m in inputs
        ),
    )
    check(
        "monitors are scoped to the tracing environment",
        all(m["modelEnvironmentName"] == "tracing" for m in inputs),
    )

    class Cfg:
        arize_region = "eu-west-1a"

    check(
        "the GraphQL endpoint follows the space's region",
        mon.graphql_url(Cfg()) == "https://app.eu-west-1a.arize.com/graphql",
        mon.graphql_url(Cfg()),
    )

    class NoRegion:
        arize_region = None

    check(
        "and falls back to the default host when unset",
        mon.graphql_url(NoRegion()) == "https://app.arize.com/graphql",
    )


def check_retrieval_accumulates() -> None:
    """A turn may search twice; the record must cover both, not just the last."""
    console.print("\n[bold]Retrieval accumulation[/bold]")
    from copilot.tools import ToolContext, search_docs

    # search_traced emits a RETRIEVER span, so it needs a tracer. Install a
    # local provider with no exporter -- this keeps the check offline while
    # still exercising the real code path rather than a stubbed one.
    import copilot.tracing as tracing

    if tracing._TRACER is None:
        from opentelemetry.sdk.trace import TracerProvider

        tracing._TRACER = TracerProvider().get_tracer("selfcheck")
        tracing._INITIALIZED = True

    ctx = ToolContext()
    first = search_docs(ctx, "What scopes can an API key have?")
    after_first = list(ctx.retrieved_doc_ids)
    second = search_docs(ctx, "How many times do you retry a failed webhook?")

    check("each search returns only its own hits", first != second and bool(first))
    check(
        "doc ids accumulate across searches",
        set(after_first) < set(ctx.retrieved_doc_ids),
        f"{after_first} -> {ctx.retrieved_doc_ids}",
    )
    check(
        "recorded context keeps the earlier search too",
        first in ctx.retrieved_context and second in ctx.retrieved_context,
    )
    ids_before = list(ctx.retrieved_doc_ids)
    search_docs(ctx, "What scopes can an API key have?")
    check("repeating a search does not duplicate ids", ctx.retrieved_doc_ids == ids_before)


def check_annotation_queue_inputs() -> None:
    """A queue that never gets created takes the human-review step with it."""
    console.print("\n[bold]Annotation queue plumbing[/bold]")
    from pathlib import Path as _Path

    source = _Path(__file__).with_name("06_annotations.py").read_text()

    # A plain dict is routed through AnnotationQueueRecordInput.from_dict, which
    # json.dumps it -- and the record source carries two datetimes, so the call
    # dies with "Object of type datetime is not JSON serializable". The typed
    # input serialises them properly.
    check(
        "the queue's record source is the typed input, not a dict",
        "AnnotationQueueSpanRecordInput(" in source,
    )
    check(
        "record_type is the upper-case enum value the schema accepts",
        'record_type="SPAN"' in source and '"record_type": "span"' not in source,
    )

    # Both failures above were swallowed into a one-line yellow warning, so the
    # step reported success for weeks while creating nothing. The reason has to
    # reach the console.
    queue_block = source.split("Queue not created")[1][:120]
    check("a failed queue creation prints why", "{exc}" in queue_block, queue_block.strip()[:60])

    # Arize 404s unless every annotator is a real user with space access, so a
    # hardcoded placeholder makes this work for nobody.
    check(
        "the annotator is resolved from the account, not hardcoded",
        "def reviewer_email(" in source and "annotator_emails=[reviewer]" in source,
    )
    check(
        "...with an env override for multi-seat accounts",
        "POC_REVIEWER_EMAIL" in source,
    )
    check(
        "the same reviewer is recorded on the annotations",
        '.updated_by": reviewer,' in source,
    )

    # A queue holds spans from one project but its name is unique per space, so
    # a fixed name 409s on the second project and leaves the new one with no
    # queue while the stale one still shows the old spans.
    check(
        "the queue name is scoped to the project",
        "def queue_name(" in source and 'name="Groundedness review"' not in source,
    )

    # The online eval task has the same failure shape, discovered on the second
    # project's tour: `find_task` matches on name space-wide, so a fixed name
    # "reuses" the first project's task, prints success, and leaves the new
    # project's spans with no online evaluator at all.
    task_source = _Path(__file__).with_name("05_online_evals.py").read_text()
    check(
        "the online task name is scoped to the project too",
        "Groundedness monitor ({settings.arize_project_name})" in task_source
        and '"Groundedness monitor",' not in task_source,
    )

    # The simulated reviewer's noise has to be a property of the span, not of
    # the interpreter. `hash()` on a str is salted per process, so the labels
    # changed on every run -- and poc/06b splits exactly these labels into train
    # and holdout to measure a template change against.
    check(
        "the label noise is a digest, not a salted hash()",
        "hashlib.sha256" in source and "abs(hash(" not in source,
    )
    from importlib import import_module

    module = import_module("06_annotations")
    ids = [f"span{i:03d}" for i in range(40)]
    check(
        "...and lands on the same spans every run",
        "".join("1" if module.is_noisy(i) else "0" for i in ids)
        == "0000000100000000000000000000010000000000",
    )


def check_judge_alignment() -> None:
    """The alignment step is only worth anything if it cannot cheat."""
    console.print("\n[bold]Judge alignment (step 06b)[/bold]")
    from pathlib import Path as _Path

    from copilot.evals import GROUNDEDNESS_TEMPLATE, build_aligned_template

    example = {
        "question": "Can I get a refund?",
        "context": "No refund policy is documented.",
        # A real answer can contain braces. The aligned template is spliced
        # together and then `.format`ted, so an unescaped one raises KeyError on
        # every subsequent grading call -- the judge would fail closed, silently.
        "answer": 'Sure: {"refund": "30 days"}',
        "label": "hallucinated",
        "reason": "Invented a window.",
    }
    aligned = build_aligned_template([example])
    check("worked examples land before the case being graded",
          aligned.index("Reviewed case 1") < aligned.index("{question}"))
    try:
        aligned.format(question="q", retrieved_context="c", answer="a")
        formats = True
    except (KeyError, IndexError, ValueError):
        formats = False
    check("a brace in an answer doesn't break the template", formats)

    unescaped = build_aligned_template([example], escape=False)
    check("escape=False leaves AX's own placeholders alone", "{{" not in unescaped)
    check("no examples means the template is untouched",
          build_aligned_template([]) == GROUNDEDNESS_TEMPLATE)

    source = _Path(__file__).with_name("06b_align_judge.py").read_text()
    # Mining few-shot examples from the rows you then score is train-on-test: it
    # reports a large gain every time and measures nothing.
    check("agreement is measured on rows the examples were not drawn from",
          "def split(" in source and "holdout" in source)
    check("...and the split is stratified, so both halves carry disagreements",
          'groupby("agreed"' in source)
    check("the result is tested, not just differenced", "mcnemar_p(" in source)
    check("too few discordant rows is called out rather than glossed over",
          "fixed + broken < 6" in source)
    # The hosted template has no retrieved-context placeholder, so it grades
    # groundedness blind. Attaching the offline judge's holdout number to that
    # version's commit message would describe a judge that was never measured.
    check("the published version doesn't borrow the offline judge's number",
          "grades_blind" in source and "Offline holdout agreement" in source)
    check("...and the discrepancy is surfaced when it applies",
          "no retrieved-context" in source)

    import importlib

    module = importlib.import_module("06b_align_judge")
    ax = importlib.import_module("05_online_evals")
    # Detecting this from the prose finds the hosted template's own sentence
    # about "documentation it retrieved" and concludes it has the documents,
    # when its only placeholders are input.value and output.value.
    check("the offline template is seen to supply context",
          module.supplies_context(GROUNDEDNESS_TEMPLATE))
    check("the hosted template is correctly seen NOT to",
          not module.supplies_context(ax.GROUNDEDNESS_TEMPLATE))


def check_session_evals() -> None:
    """A session score written to every span silently weights by conversation length."""
    console.print("\n[bold]Session-level evaluation (step 04b)[/bold]")
    import importlib
    from pathlib import Path

    import pandas as pd

    module = importlib.import_module("04b_session_evals")

    turns = pd.DataFrame(
        [
            {"session_id": "s1", "span_id": "a", "question": "q1", "answer": "a1",
             "tool_calls": "", "is_failure": False, "start_time": 1},
            {"session_id": "s1", "span_id": "b", "question": "q2", "answer": "a2",
             "tool_calls": "escalate_ticket", "is_failure": False, "start_time": 2},
            {"session_id": "s2", "span_id": "c", "question": "q3", "answer": "a3",
             "tool_calls": "", "is_failure": True, "start_time": 3},
            {"session_id": "", "span_id": "d", "question": "q4", "answer": "a4",
             "tool_calls": "", "is_failure": False, "start_time": 4},
        ]
    )
    turns["turn_failed"] = turns["is_failure"]
    built = module.build_transcripts(turns)
    check("one row per session, not per turn", len(built) == 2, f"got {len(built)}")

    # Nothing documents the export as chronological, and `spans.list()` is
    # explicitly descending. Fed the reverse order, an unsorted implementation
    # hands the judge a backwards conversation and anchors the verdict on the
    # opening turn -- both of which produce a plausible-looking score.
    shuffled = module.build_transcripts(turns.iloc[::-1].reset_index(drop=True))
    s1_shuffled = shuffled[shuffled["session_id"] == "s1"].iloc[0]
    check("turn order survives a reversed input", s1_shuffled["span_id"] == "b",
          s1_shuffled["span_id"])
    check("...and so does the transcript",
          s1_shuffled["transcript"].index("q1") < s1_shuffled["transcript"].index("q2"))
    check("spans with no session id are dropped", "" not in set(built["session_id"]))
    s1 = built[built["session_id"] == "s1"].iloc[0]
    check("the verdict hangs on the session's last span", s1["span_id"] == "b", s1["span_id"])
    check("the transcript carries both turns", "q1" in s1["transcript"] and "q2" in s1["transcript"])
    check("tool calls are visible to the judge", "escalate_ticket" in s1["transcript"])
    check("escalation is detected across the session", bool(s1["escalated"]))
    check("turn-level failures are carried through for the comparison",
          int(built[built["session_id"] == "s2"].iloc[0]["turn_failures"]) == 1)

    # The headline claim -- "failed as a whole, no turn-level failure" -- is only
    # meaningful if it clears every turn-level signal. Step 03's heuristics flag
    # 10 turns on current traffic where step 04's evaluators flag 38, so scoring
    # it on the heuristics alone counts sessions the judge already condemned.
    source = Path(__file__).with_name("04b_session_evals.py").read_text()
    check("the silent-session count consults step 04's verdicts too",
          "TURN_FAILURE_EVALS" in source and "04_evals.parquet" in source)
    check("...and reports which signals it cleared",
          "verdict_sources" in source)
    check("a turn is counted failed if either signal fires",
          'merged["turn_failed"] | (merged[score_cols] < 1.0)' in source)

    graded = pd.DataFrame(
        [
            # Heuristic-clean, but the judge flagged it: must not read as silent.
            {"span_id": "a", "is_failure": False, "eval.groundedness.score": 0.0},
            {"span_id": "b", "is_failure": False, "eval.groundedness.score": 1.0},
        ]
    )
    graded["context.span_id"] = graded["span_id"]
    combined = graded["is_failure"] | (graded[["eval.groundedness.score"]] < 1.0).any(axis=1)
    check("an eval-flagged, heuristic-clean turn counts as a failure",
          bool(combined.iloc[0]) and not bool(combined.iloc[1]))
    # NaN means never graded, which is not evidence of a failure.
    ungraded = pd.DataFrame([{"eval.groundedness.score": float("nan")}])
    check("an ungraded turn is not counted as a failure",
          not bool((ungraded[["eval.groundedness.score"]] < 1.0).any(axis=1).iloc[0]))

    from copilot.evals import SESSION_CHOICES, parse_verdict

    check("three ordered outcomes, so 'unhelpful' and 'harmful' differ",
          len(SESSION_CHOICES) == 3 and SESSION_CHOICES["unresolved"] > SESSION_CHOICES["frustrated"])
    label, _, _ = parse_verdict("unresolved\nNothing was fixed.", SESSION_CHOICES)
    check("a multi-word label parses on its own scale", label == "unresolved", label)


def check_backfill_spans() -> None:
    """Backfilled spans must be reproducible and must not pollute the live project."""
    console.print("\n[bold]Direct span ingestion (step 02b)[/bold]")
    import importlib
    from pathlib import Path as _Path

    module = importlib.import_module("02b_log_spans")

    spans, evals = module.build_history(module.HISTORIC_TICKETS, 30)
    required = {"context.trace_id", "context.span_id", "name"}
    check("the required OpenInference columns are present", required <= set(spans.columns))
    check("ids are stable across builds, so a re-run replaces rather than doubles",
          spans["context.span_id"].tolist()
          == module.build_history(module.HISTORIC_TICKETS, 30)[0]["context.span_id"].tolist())
    check("span ids are unique", spans["context.span_id"].is_unique)
    roots = spans[spans["parent_id"] == ""]
    check("every trace has exactly one root", len(roots) == len(module.HISTORIC_TICKETS))
    children = spans[spans["parent_id"] != ""]
    check("children point at a root that exists",
          set(children["parent_id"]) <= set(roots["context.span_id"]))
    check("evals join to the root spans only",
          set(evals["context.span_id"]) == set(roots["context.span_id"]))
    check("start precedes end on every span", bool((spans["end_time"] > spans["start_time"]).all()))

    source = _Path(__file__).with_name("02b_log_spans.py").read_text()
    # Mixing backfilled rows into the analysed project would change every count
    # step 03 reports, with nothing marking them as a different source.
    check("it targets its own project, not the one step 03 analyses",
          '-backfill' in source)
    check("read-back polls rather than reading once",
          "POLL_ATTEMPTS" in source)


def check_span_metadata_enrichment() -> None:
    """Findings that stay in a local parquet leave the trace view no better off."""
    console.print("\n[bold]Span metadata enrichment (step 03)[/bold]")
    from pathlib import Path

    source = Path(__file__).with_name("03_query_spans.py").read_text()
    check("the failure classification is written back to the spans",
          "update_metadata(" in source)
    check("...under attributes.metadata.*, which AX converts to a merge patch",
          "attributes.metadata.failure_mode" in source)
    # "" and "unset" are the same thing in a UI filter, and "no failures" is a
    # finding rather than an absence.
    check('a clean turn is tagged "none", not empty', '.replace("", "none")' in source)
    check("it can be turned off without skipping the export", "skip_metadata" in source)
    # Without this column step 04b cannot order a conversation at all.
    check("start_time is carried through for the session judge",
          '"start_time": find_col(' in source and '"start_time": span.get(' in source)


def check_dataset_lifecycle() -> None:
    """A dataset that grows by a full copy per run silently triples every experiment."""
    console.print("\n[bold]Dataset lifecycle (step 07)[/bold]")
    from pathlib import Path

    source = Path(__file__).with_name("07_dataset.py").read_text()
    check("re-runs append only what is new", "fresh = [e for e in examples" in source)
    check("human verdicts reach the dataset as fields", "update_examples(" in source)
    check("...and are attempted as annotations too", "annotate_examples(" in source)
    # The platform rejects the whole batch if a score is sent for a categorical
    # config, and step 06 creates a categorical one.
    check("no score is sent with a categorical annotation",
          "AnnotationInput(name=ANNOTATION_NAME, label=verdict" in source
          and "score=float(row.get" not in source)
    check("a failed merge exits non-zero instead of printing a summary",
          "merge_failed" in source)
    # Both dataset endpoints cap a request at 1000 records, and blowing the cap
    # 4xxs partway through, leaving the dataset half-updated.
    check("dataset writes are batched under the server's 1000-record cap",
          "BATCH_LIMIT = 1000" in source and source.count("chunked(") == 3)


def check_experiment_arms() -> None:
    """Each arm has to differ from its reference by exactly one thing."""
    console.print("\n[bold]Experiment arms (step 08)[/bold]")
    import inspect
    from pathlib import Path

    source = Path(__file__).with_name("08_experiments.py").read_text()
    check("the model is a variable, not a constant", "compare_model" in source)
    check("run_turn accepts the override", "build_task(settings, version, model)" in source)
    # v2-on-flash against v1-on-pro changes the prompt and the model at once,
    # and no test can attribute the difference to either.
    check("the model arm is measured against the candidate prompt, not the baseline",
          "candidate,\n            )" in source or "reference" in source)
    check("both arms go through the same paired test",
          source.count("compare(reference, label") == 1 and "def compare(" in source)
    # COPILOT_AGENT_MODEL can point the baseline arms at flash, in which case a
    # flash comparison arm changes nothing and presents run-to-run noise as a
    # model difference.
    check("an arm that wouldn't change the model is skipped",
          "compare_model == AGENT_MODEL" in source)
    check("...and says why rather than silently dropping it",
          "Skipping the model arm" in source)

    from copilot.agent import run_turn

    check("run_turn takes a model override", "model" in inspect.signature(run_turn).parameters)


def check_prompts_and_tools() -> None:
    console.print("\n[bold]Prompts and tool schemas[/bold]")
    from copilot.prompts import V1, V2
    from copilot.tools import tools_for

    check("v2 is substantially longer than v1", len(V2) > len(V1) * 3)
    for term in ("ground", "escalate", "150 words"):
        check(f"v2 addresses {term!r}", term.lower() in V2.lower())
    for term in ("ground", "escalate"):
        check(f"v1 does NOT address {term!r} (it's the baseline)", term.lower() not in V1.lower())

    from copilot.tools import description_of

    v1_tools, v2_tools = tools_for("v1"), tools_for("v2")
    check("both versions expose 3 tools", len(v1_tools) == len(v2_tools) == 3)
    check(
        "v2 tool descriptions are far more specific",
        sum(len(description_of(t)) for t in v2_tools)
        > sum(len(description_of(t)) for t in v1_tools) * 5,
    )
    # OpenAI function-calling shape, which is what DeepSeek's endpoint accepts.
    for tools, version in ((v1_tools, "v1"), (v2_tools, "v2")):
        ok = all(
            t.get("type") == "function"
            and {"name", "description", "parameters"} <= set(t.get("function", {}))
            and t["function"]["parameters"].get("type") == "object"
            for t in tools
        )
        check(f"{version} tool schemas are well-formed", ok)


def check_dataframe_contracts() -> None:
    """The highest-value offline check: the SDK validates these with regexes."""
    console.print("\n[bold]Arize dataframe contracts[/bold]")
    from arize.spans.columns import ANNOTATION_COLUMN_PATTERN, EVAL_COLUMN_PATTERN

    import re

    eval_re = re.compile(EVAL_COLUMN_PATTERN)
    annot_re = re.compile(ANNOTATION_COLUMN_PATTERN)

    from importlib import import_module

    offline = import_module("04_offline_evals")

    eval_names = [
        offline.GROUNDEDNESS,
        offline.TOOL_SELECTION,
        offline.ESCALATION,
        offline.CONCISENESS,
    ]
    for name in eval_names:
        cols = [f"eval.{name}.{suffix}" for suffix in ("label", "score", "explanation")]
        ok = all(eval_re.match(c) for c in cols)
        check(f"eval columns for `{name}` match the SDK pattern", ok, cols[0])

    annotations = import_module("06_annotations")
    annot_cols = [
        f"annotation.{annotations.ANNOTATION_NAME}.{s}"
        for s in ("label", "score", "text", "updated_by", "updated_at")
    ]
    check(
        f"annotation columns for `{annotations.ANNOTATION_NAME}` match the SDK pattern",
        all(annot_re.match(c) for c in annot_cols),
        annot_cols[0],
    )


def check_targets_ax_not_phoenix() -> None:
    """This POC exercises Arize AX, the hosted platform -- not Arize Phoenix.

    The two share OpenInference conventions and a lot of vocabulary, so it is
    easy for a project to drift from one to the other without anyone noticing.
    These checks pin the distinction down:

      * traces must leave via arize.otel, aimed at Arize's OTLP collector
      * every platform operation must go through ArizeClient
      * nothing may import phoenix.otel, phoenix.trace, or run a Phoenix server

    `phoenix.evals` is the deliberate exception. It is a standalone evaluation
    library, and the AX docs themselves prescribe it for code-based evals
    (arize.com/docs/ax/evaluate/run-evals-on-traces) -- scores are computed
    locally, then written to AX with client.spans.update_evaluations().
    """
    console.print("\n[bold]Targets AX, not Phoenix[/bold]")
    import os
    from pathlib import Path

    from arize.otel import Endpoint

    check(
        "trace exporter defaults to the Arize collector",
        Endpoint.ARIZE.value == "https://otlp.arize.com/v1",
        Endpoint.ARIZE.value,
    )
    # A stale value here would silently send the whole tour somewhere else.
    override = os.getenv("ARIZE_COLLECTOR_ENDPOINT") or os.getenv("ARIZE_OTLP_ENDPOINT")
    check(
        "no collector-endpoint override pointing away from Arize",
        not override or "arize.com" in override,
        f"set to {override!r}",
    )

    root = Path(__file__).resolve().parents[1]
    # This file is excluded: it names the banned tokens in order to look for them.
    sources = [
        p
        for p in (*(root / "src" / "copilot").glob("*.py"), *(root / "poc").glob("*.py"))
        if p.name != Path(__file__).name
    ]
    banned = ("phoenix.otel", "phoenix.trace", "phoenix.session", "px.launch_app")
    offenders = [
        f"{p.name}:{token}"
        for p in sources
        for token in banned
        if token in p.read_text(encoding="utf-8")
    ]
    check("no Phoenix tracing/server imports", not offenders, str(offenders))

    tracing_src = (root / "src" / "copilot" / "tracing.py").read_text(encoding="utf-8")
    check(
        "tracing bootstrap registers via arize.otel",
        "from arize.otel import register" in tracing_src,
    )


def check_region_wiring() -> None:
    """Region has to reach the exporter *and* the platform client.

    A key from one region is not valid in another, so a half-applied region is
    the difference between "tracing works but every export 401s" and a working
    tour. These checks pin the mapping without needing credentials.
    """
    console.print("\n[bold]Region wiring[/bold]")
    import os

    from arize.regions import Region

    from copilot.config import DEFAULT_OTLP_ENDPOINT, _resolve_otlp_endpoint

    configured = os.getenv("ARIZE_REGION") or None
    check(
        f"ARIZE_REGION is a value the SDK accepts ({configured or 'unset → US default'})",
        configured is None or configured in {r.value for r in Region},
        f"{configured!r} is not one of {sorted(r.value for r in Region)}",
    )

    # Only assert the mapping when nothing is overriding it, or the override
    # (legitimately) wins and these would fail for the wrong reason.
    explicit = os.getenv("ARIZE_COLLECTOR_ENDPOINT") or os.getenv("ARIZE_OTLP_ENDPOINT")
    if explicit:
        console.print(f"  [dim]— endpoint pinned explicitly to {explicit}; mapping not asserted[/dim]")
    else:
        check(
            "eu-west-1a resolves to the EU collector",
            _resolve_otlp_endpoint("eu-west-1a") == "https://otlp.eu-west-1a.arize.com/v1",
            _resolve_otlp_endpoint("eu-west-1a"),
        )
        check(
            "no region resolves to the US collector",
            _resolve_otlp_endpoint(None) == DEFAULT_OTLP_ENDPOINT,
            _resolve_otlp_endpoint(None),
        )


def check_langgraph_engine() -> None:
    """The second engine: same logic through LangGraph/LangChain.

    Everything here runs offline -- the graph compiles, tools bind, and the
    dispatch fires without a single network call, so a live failure means
    "model/platform", not "the wiring is wrong".
    """
    console.print("\n[bold]LangGraph engine (COPILOT_IMPL=langgraph)[/bold]")
    import os
    from pathlib import Path

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    import copilot.agent as agent_mod
    from copilot import graph_agent as ga
    from copilot.agent import MAX_TOOL_ITERATIONS, TurnResult
    from copilot.config import Settings, load_settings
    from copilot.tools import TOOLS_V1, TOOLS_V2

    nodes = set(ga.build_graph().compile().get_graph().nodes)
    check(
        "the graph compiles with the copilot's four nodes",
        {"classify", "agent", "tools", "give_up"} <= nodes,
        f"nodes: {sorted(nodes)}",
    )

    # The seeded tool mis-routing lives in the v1 descriptions, so they must
    # reach the model byte for byte -- bind_tools takes the dicts as-is.
    llm = ga._ChatDeepSeek(
        model="deepseek-v4-pro", api_key="offline", api_base="https://example.invalid/v1"
    )
    check("v1's deliberately-vague tool schemas bind verbatim",
          llm.bind_tools(TOOLS_V1).kwargs["tools"] == TOOLS_V1)
    check("...and v2's fixed ones too",
          llm.bind_tools(TOOLS_V2).kwargs["tools"] == TOOLS_V2)

    # DeepSeek 400s a tool loop whose assistant messages come back without
    # their CoT; ChatDeepSeek captures it but never re-sends it, so the echo
    # lives in our subclass and is asserted here.
    tool_call = [{"name": "search_docs", "args": {"query": "q"}, "id": "call-1"}]
    payload = llm._get_request_payload(
        [
            HumanMessage("q"),
            AIMessage(content="", tool_calls=tool_call,
                      additional_kwargs={"reasoning_content": "COT"}),
            ToolMessage(content="result", tool_call_id="call-1"),
        ]
    )
    assistants = [m for m in payload["messages"] if m["role"] == "assistant"]
    check("tool-loop requests echo reasoning_content back",
          bool(assistants) and assistants[0].get("reasoning_content") == "COT")
    plain = llm._get_request_payload(
        [HumanMessage("q"), AIMessage(content="hi", additional_kwargs={"reasoning_content": "COT"})]
    )
    check("...but plain assistant messages do not carry it",
          all("reasoning_content" not in m for m in plain["messages"]))

    # The loop cap is enforced in the routing function, not recursion_limit.
    looping = {
        "messages": [AIMessage(content="", tool_calls=tool_call)],
        "agent_calls": 1, "intent": "other", "error": None,
    }
    capped = {**looping, "agent_calls": MAX_TOOL_ITERATIONS}
    finished = {**looping, "messages": [AIMessage(content="done")]}
    check(
        "the tool loop routes loop/cap/end correctly",
        ga.decide_next(looping) == "tools"
        and ga.decide_next(capped) == "give_up"
        and ga.decide_next(finished) == "end",
    )

    # Dispatch: settings.impl decides the engine, at call time, per turn.
    dummy = Settings(
        arize_api_key="x", arize_space_id="x", arize_space_name="x",
        arize_project_name="x", arize_region=None, arize_otlp_endpoint="x",
        deepseek_api_key="x", deepseek_base_url="x",
        arize_ai_integration_id=None, prompt_version="v1", impl="langgraph",
    )
    sentinel = TurnResult(question="q", answer="from-langgraph")
    real_run_turn = ga.run_turn
    try:
        ga.run_turn = lambda question, **kwargs: sentinel
        dispatched = agent_mod.run_turn("q", settings=dummy)
    finally:
        ga.run_turn = real_run_turn
    check("agent.run_turn dispatches to the LangGraph engine", dispatched is sentinel)

    # The -lg project suffix keeps the engines' traffic apart.
    saved = {k: os.environ.get(k) for k in
             ("ARIZE_API_KEY", "ARIZE_SPACE_ID", "ARIZE_SPACE_NAME", "DEEPSEEK_API_KEY",
              "ARIZE_PROJECT_NAME", "COPILOT_IMPL")}
    try:
        os.environ.update(
            ARIZE_API_KEY="x", ARIZE_SPACE_ID="x", ARIZE_SPACE_NAME="x",
            DEEPSEEK_API_KEY="x", ARIZE_PROJECT_NAME="proj",
        )
        os.environ["COPILOT_IMPL"] = "langgraph"
        lg_project = load_settings().arize_project_name
        os.environ["COPILOT_IMPL"] = "sdk"
        sdk_project = load_settings().arize_project_name
    finally:
        for key, value in saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value
    check(
        "COPILOT_IMPL=langgraph writes to its own -lg project",
        lg_project == "proj-lg" and sdk_project == "proj",
        f"langgraph={lg_project!r} sdk={sdk_project!r}",
    )

    # One instrumentor, never both -- two would double-count tokens and cost.
    source = (Path(__file__).parent.parent / "src" / "copilot" / "tracing.py").read_text()
    check(
        "tracing instruments exactly one SDK per engine",
        'if settings.impl == "langgraph"' in source
        and "LangChainInstrumentor" in source
        and source.index("LangChainInstrumentor") < source.index("OpenAIInstrumentor"),
    )


def check_sdk_surface() -> None:
    console.print("\n[bold]SDK surface[/bold]")
    try:
        from arize.client import ArizeClient

        client = ArizeClient(api_key="offline-selfcheck")
        for resource in (
            "datasets",
            "experiments",
            "evaluators",
            "tasks",
            "spans",
            "prompts",
            "annotation_configs",
            "annotation_queues",
            "ai_integrations",
            "projects",
        ):
            check(f"client.{resource} exists", hasattr(client, resource))
    except Exception as exc:  # noqa: BLE001
        check("arize SDK imports", False, str(exc))

    for module, symbol in [
        ("arize.experiments", "EvaluationResult"),
        ("arize.evaluators.types", "TemplateConfig"),
        ("arize.evaluators.types", "EvaluatorLlmConfig"),
        ("arize.evaluators.types", "CustomCodeConfig"),
        ("arize.tasks.types", "TaskType"),
        ("arize.tasks.types", "TaskEvaluatorInput"),
        ("arize.prompts.types", "LLMMessage"),
        ("arize.annotation_configs.types", "CategoricalAnnotationValue"),
        ("arize.otel", "register"),
        ("openinference.instrumentation.openai", "OpenAIInstrumentor"),
        ("arize.ai_integrations.types", "AiIntegrationProvider"),
        ("phoenix.evals", "create_classifier"),
    ]:
        try:
            mod = __import__(module, fromlist=[symbol])
            check(f"{module}.{symbol}", hasattr(mod, symbol))
        except Exception as exc:  # noqa: BLE001
            check(f"{module}.{symbol}", False, str(exc))


def main() -> None:
    console.print("\n[bold cyan]Arize AX POC — offline self-check[/bold cyan]")
    console.print("[dim]No credentials or network required.[/dim]")

    check_knowledge_base()
    check_questions()
    check_code_evaluators()
    check_retrieval_accumulates()
    check_experiment_statistics()
    check_judge_verdict_parsing()
    check_prompt_hub_plumbing()
    check_monitor_metrics()
    check_annotation_queue_inputs()
    check_judge_alignment()
    check_session_evals()
    check_backfill_spans()
    check_span_metadata_enrichment()
    check_dataset_lifecycle()
    check_experiment_arms()
    check_prompts_and_tools()
    check_dataframe_contracts()
    check_targets_ax_not_phoenix()
    check_region_wiring()
    check_langgraph_engine()
    check_sdk_surface()

    console.print()
    if FAILURES:
        table("Failed checks", ["check"], [[f] for f in FAILURES])
        console.print(f"\n[bold red]{len(FAILURES)} check(s) failed.[/bold red]\n")
        raise SystemExit(1)
    console.print("[bold green]All checks passed.[/bold green]")
    console.print(
        "\nNext: copy .env.example to .env, fill in credentials, then\n"
        "  [bold]uv run python poc/01_trace.py[/bold]\n"
    )


if __name__ == "__main__":
    main()

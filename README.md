# Arize AX POC — Nimbus Support Copilot

A deliberately imperfect support agent, built to exercise the main features of
[Arize AX](https://arize.com/docs/ax) end to end: **instrument → observe →
evaluate → improve**.

The agent is the instrument, not the point. It is a RAG + tool-use copilot for a
fictional SaaS ("Nimbus"), wired so that every OpenInference span kind appears in
the trace tree, and seeded with four failure modes that AX's evaluators are built
to catch. The tour then attempts to fix those failures and puts the fix to a
statistical test.

That test currently says the fix is **not** proven — see
[the acceptance criterion](#the-tour). That is a real result, not a broken repo,
and arguably the more useful demonstration: the loop's job is to tell you when
you have not improved anything.

---

## AX, not Phoenix

The platform under test is **Arize AX**, the hosted product. Arize Phoenix — the
open-source sibling — is not being demonstrated, and nothing here runs a Phoenix
server or exports to one.

- Traces leave through `arize.otel.register()` to Arize's collector
  (`https://otlp.arize.com/v1`), never `phoenix.otel`.
- Every platform operation — spans, datasets, experiments, evaluators, tasks,
  annotations, prompts — goes through `arize.client.ArizeClient`.
- Monitors use the Arize GraphQL API.

The one exception is `phoenix.evals` in step 04, and it is deliberate:
**Phoenix Evals is a standalone evaluation library, not the Phoenix platform**,
and [the AX docs prescribe it](https://arize.com/docs/ax/evaluate/run-evals-on-traces)
for code-based evals. Scores are computed locally and then written to AX with
`client.spans.update_evaluations()`. Step 05 shows the same judgement as a
*native AX* online evaluator running on Arize's infrastructure — the two exist
side by side on purpose, because that offline→online move is the point.

`make check` asserts all of this, so the POC can't quietly drift.

---

## Why the agent is bad on purpose

A POC where the agent answers everything correctly demonstrates nothing: the
evaluators all return 1.0, the dataset is uninteresting, and the experiment shows
no delta. So four failures are built in.

| Failure mode | How it's seeded | Caught by |
|---|---|---|
| **Hallucination** | The KB has no refund page. Retrieval returns *adjacent* chunks (cancellation, billing), and the v1 prompt never says "stay in the context" — so the model invents a refund window. | LLM-as-judge (groundedness) |
| **Verbosity** | v1 says "be thorough" with no length guidance. | Code evaluator |
| **Wrong tool** | v1's tool descriptions are one-liners (`"Get order info."`), so order questions should get routed to `search_docs`. | Code evaluator on the trajectory |
| **Missing escalation** | v1 never mentions escalation, so blocked and angry users should never reach a human. | Code evaluator |

### The bottom two traps no longer fire

Measured against `deepseek-v4-pro` with thinking on: **8/8** order questions
called `lookup_order` and **5/5** escalation questions called `escalate_ticket`.
The model reasons past the deliberately useless tool descriptions every time.
These traps were designed for a weaker, non-thinking model, and a stronger one
walks through them.

So **the acceptance criterion is groundedness and conciseness, not escalation** —
v1 already scores 1.00 on escalation and there is no headroom to improve. The
first two rows still produce plenty of signal: the judge flags roughly half of
all v1 answers as ungrounded. If you want the full four-mode demonstration back,
the traps need recalibrating for a thinking model (making v1's tool descriptions
actively misleading rather than merely vague), not just re-running.

This is worth dwelling on, because it *is* the point of the tooling: the seeded
expectation was wrong, and the evaluators are what revealed it.

---

## Setup

```bash
uv sync
cp .env.example .env      # fill in Arize + DeepSeek credentials
make check                # offline self-check: no credentials needed
```

`make check` validates the corpus, the fixture, the evaluators, and that the
dataframes match the Arize SDK's column contracts — so a later failure means
"credentials or platform", not "the repo is broken".

You need an Arize space (`ARIZE_API_KEY`, `ARIZE_SPACE_ID`, `ARIZE_SPACE_NAME`)
and a `DEEPSEEK_API_KEY`. One optional extra is called out at step 05.

**If your space is not in the US, set `ARIZE_REGION`** (e.g. `eu-west-1a`).
Arize keys are region-scoped and a cross-region key fails to authorize rather
than redirecting — on the trace exporter that surfaces as the fairly opaque
`unable to validate authorization from span`. Region has to reach two places:
the OTel collector URL and the platform client. `ARIZE_REGION` drives both, so
prefer it over setting an endpoint by hand; `ARIZE_COLLECTOR_ENDPOINT` (or
`ARIZE_OTLP_ENDPOINT`) still wins if you need to pin one explicitly. Each
script prints the region it resolved in its banner.

---

## The tour

Run in order. Each script prints what to go and look at in the AX UI.

| Step | Command | What it demonstrates | AX docs |
|---|---|---|---|
| 00 | `make check` | Offline self-check | — |
| 01 | `poc/01_trace.py` | Tracing, span kinds, **sessions**, users, token cost | [tracing](https://arize.com/docs/ax/get-started/get-started-tracing) · [sessions](https://arize.com/docs/ax/instrument/set-up-sessions) · [costs](https://arize.com/docs/ax/instrument/track-costs) |
| 02 | `poc/02_customize_traces.py` | Custom attributes, metadata, tags, **PII redaction** | [customize](https://arize.com/docs/ax/instrument/customize-your-traces) · [masking](https://arize.com/docs/ax/instrument/mask-and-redact-data) |
| 02b | `poc/02b_log_spans.py` | **Backfill spans without OTel** — `spans.log()` into its own project | [spans resource](https://arize.com/docs/api-clients/python/version-8/client-resources/spans) |
| 03 | `poc/03_query_spans.py` | Export spans, reconstruct trajectories, **find the failures**, tag them back onto the spans | [view traces](https://arize.com/docs/ax/observe/tracing/view-and-manage-traces) |
| 04 | `poc/04_offline_evals.py` | LLM-as-judge + code evals, logged back onto spans | [evals on traces](https://arize.com/docs/ax/evaluate/run-evals-on-traces) |
| 04b | `poc/04b_session_evals.py` | **Grade whole conversations**, not turns | [sessions](https://arize.com/docs/ax/observe/tracing/sessions-and-users) |
| 05 | `poc/05_online_evals.py` | Evaluators + **continuous tasks** (sampling, filters) | [online judge](https://arize.com/docs/ax/concepts/evaluators/online-llm-as-judge) · [code evals](https://arize.com/docs/ax/concepts/evaluators/online-code-evaluators) |
| 06 | `poc/06_annotations.py` | Annotation config, **review queue**, judge-vs-human agreement | [human review](https://arize.com/docs/ax/evaluate/human-review) · [align evals](https://arize.com/docs/ax/evaluate/align-evals-to-human-feedback) |
| 06b | `poc/06b_align_judge.py` | **Align the judge** to human labels, prove it on a holdout, version the evaluator | [align evals](https://arize.com/docs/ax/evaluate/align-evals-to-human-feedback) |
| 07 | `poc/07_dataset.py` | Dataset from failing traces (+ a control group), with the human review folded back in | [build a dataset](https://arize.com/docs/ax/improve/build-a-dataset) |
| 08 | `poc/08_experiments.py` | **v1 vs v2, and pro vs flash, on identical inputs** — the payoff | [experiments](https://arize.com/docs/ax/improve/set-up-an-experiment) |
| 09 | `poc/09_prompt_hub.py` | Publish, label `production`, load at runtime | [prompt hub](https://arize.com/docs/ax/concepts/prompts/prompt-hub) |
| 10 | `poc/10_monitors.py` | Monitors, alerting, dashboards | [monitoring](https://arize.com/docs/ax/observe/production-monitoring) |

Or in phases:

```bash
make trace      # 01-02
make backfill   # 02b   (own project; not part of `make all`)
make observe    # 03
make evaluate   # 04, 04b, 05, 06, 06b
make improve    # 07-09
make monitor    # 10
make all        # everything, in order
```

Lettered steps are side paths off the step they follow. `02b` is independent of
the rest of the tour and writes to its own project; `04b` and `06b` read what
the steps before them produced.

**The acceptance criterion is step 08, and it does not hold up across runs.**
The most recent tour, 35 examples against `deepseek-v4-pro`:

| evaluator | v1 | v2 | delta | rows changed | McNemar |
|---|---|---|---|---|---|
| `answers_from_context` | 1.00 | 1.00 | +0.00 ≈ | 0↑ 0↓ | p=1.000 |
| `conciseness` | 0.77 | 0.89 | +0.11 ≈ | 4↑ 0↓ | p=0.125 |
| `groundedness` | 0.49 | 0.54 | +0.06 ≈ | 3↑ 1↓ | p=0.625 |

Nothing clears p<0.05. An earlier tour on the same code did — groundedness
0.43 → 0.63, 7↑ 0↓, **p=0.016** — and four tours now read 0.48→0.52,
0.55→0.52, 0.43→0.63 and 0.49→0.54.

Those are *not* four runs of one experiment: each tour regenerates traffic in
step 01 and rebuilds the dataset in step 07, so the inputs differ every time.
But that is the point rather than an excuse. **One significant result out of
four, on a rebuilt dataset each time, is what a real improvement of this size
looks like at n=35 — indistinguishable from none.** Leading with the run that
won would be cherry-picking; the honest summary is that v2 probably helps a
little and this dataset cannot show it.

This is exactly why step 08 tests significance (paired McNemar, since every arm
answers the same inputs) rather than trusting a delta. An earlier version of the
script counted any positive delta as a win, and duly declared victory on noise —
it would have called three of these four tours a success.

Conciseness moves the same direction in run after run with 4↑ 0↓ and still can't
be called: roughly 5–6 unanimous flips are needed for p<0.05 at this n. That is
the shape of an "obvious" improvement that isn't yet evidence.

### The model arm

Same run, same dataset, holding the v2 prompt fixed and swapping
`deepseek-v4-pro` for `deepseek-v4-flash`:

| evaluator | v2 (pro) | v2 (flash) | delta | rows changed | McNemar |
|---|---|---|---|---|---|
| `answers_from_context` | 1.00 | 1.00 | +0.00 ≈ | 0↑ 0↓ | p=1.000 |
| `conciseness` | 0.89 | 1.00 | +0.11 ≈ | 4↑ 0↓ | p=0.125 |
| `groundedness` | 0.54 | 0.63 | +0.09 ≈ | 3↑ 0↓ | p=0.250 |

The cheaper model did not score worse on anything — it scored nominally
*better* on two of three, none significantly. `flash` is roughly 6× smaller on
active parameters.

Read that as **"no measurable quality cost on these 35 examples"**, not as
"flash is as good" and certainly not as "flash is better". At this sample size
the test can only detect a large drop; a real 5-point regression would sail
through undetected. It is a reason to run the comparison on a dataset big enough
to matter before moving production traffic, which is the point of having the arm
at all — a prompt-only experiment never raises the question.

The dataset is also selected *on failures* (23 failing turns + 12 controls), so
it deliberately over-samples the cases v2 targets. That is the right shape for
detecting a fix and the wrong shape for estimating production groundedness.

**Two caveats that the tour itself hands you, and that a green checkmark would
otherwise bury.**

*The metric that improved is one step 06 says not to trust yet.* Judge-vs-human
agreement came out at 35–50%, almost all of it the judge flagging answers the
human didn't. Step 08 shows v2 scoring better under that judge; it does not
show v2 hallucinating less.

Step 06b now attempts the fix rather than only naming it — and reports that it
could not certify one. Across three runs few-shot alignment moved held-out
agreement up, down and nowhere, never approaching significance. **The caveat
stands.** What changed is that there is now a measurement saying so, on rows the
alignment never saw, instead of an unexamined assumption in either direction.

*Two judges disagree wildly on the same spans.* Step 04's offline Phoenix judge
scores the 38 turns at mean 0.47; the AX-hosted online evaluator from step 05
scores the very same spans at 0.11 — visible side by side in step 10's metrics
table, since the case difference keeps them in separate columns. Same model,
same spans, different template and harness. Whichever you wire a monitor to is
the one that defines "groundedness" for your alerts.

To harden the result: widen the dataset (more traffic in step 01 → more graded
failures in step 04 → more rows in step 07, which raises the detectable effect
size), and collect more human labels — step 06b shows the alignment measurement
is label-starved long before it is technique-starved.

Step 07 builds its dataset from step 04's **eval verdicts**, not step 03's
heuristics. The heuristics are keyword-narrow — `check_ungrounded` only fires on
refund phrasing and found 2 hallucinations in 38 turns, where the judge grading
every answer flagged 20. Run step 04 before step 07 or you get the narrow set.

---

## What the lettered steps add

These cover the parts of AX the numbered tour walks past, and two of them exist
specifically to close gaps the numbered tour leaves open.

### 06b — aligning the judge, and proving it

Step 06 measures judge-vs-human agreement and stops. Step 08 then declares a
winner using that same judge. Step 06b closes the loop: it takes the cases where
the judge and the reviewer disagreed, turns them into worked examples inside the
judge's own template, and re-measures agreement **on rows the examples were not
drawn from**. Mining examples from the rows you then score is train-on-test — it
shows a large gain every time and means nothing — so the labelled rows are split
in half, stratified so both halves carry disagreements.

The corrections are picked to cover the judge's error modes in proportion to how
often each occurs, not by taking the first few. A judge corrected on only half
of how it is wrong tends to lurch the other way; an earlier version did exactly
that and made agreement *worse*.

The honest result on this fixture: 38 labelled rows, 20 disagreements, a 19-row
holdout — and **no consistent effect**. Three runs of the identical step over
the identical labels gave +2/−0 (37% → 47%), +1/−1 (37% → 37%) and +0/−0
(42% → 42%). Every one of them p ≥ 0.5.

A two-sided exact test cannot reach p<0.05 with fewer than six discordant rows
however lopsided they are, so the step says so rather than letting a green arrow
imply otherwise. The binding constraint is the number of human labels, not the
template.

Note what the *unaligned* judge did across those same three runs, over the same
spans and the same labels: 37%, 37%, 42% — and 47% and 60% on two earlier ones.
**The judge's run-to-run variance is larger than the effect being measured.**
That single observation is the whole argument for a held-out significance test
instead of a before/after number, and it is why quoting the run that looked best
would have been the easiest mistake in this repo to make. The first draft of
this section did exactly that.

It also publishes the aligned template as a new version of the AX `Groundedness`
evaluator. The hosted template is *not* the local one — it addresses
`{input.value}` rather than named placeholders — so the examples are spliced
into whatever AX currently holds, read back from the platform. That carries
forward any edit made in the UI, and means the braces must not be escaped on
that path even though they must be on the local one.

More importantly, **the hosted template has no placeholder for the retrieved
documents at all**: its only two are `{input.value}` and `{output.value}`, so it
grades groundedness without ever seeing the source. That is almost certainly why
step 10 shows the offline judge at 0.47 and this one at 0.11 on identical spans.
The worked examples transfer between them; the holdout number does not, and the
commit message says so rather than implying the published version was the thing
measured. Detecting this needs the placeholder list, not the prose — the hosted
template contains the phrase "documentation it retrieved", which is what a
keyword check finds just before drawing the opposite conclusion.

### 04b — grading conversations instead of turns

Everything else grades one answer at a time, which is what most eval tooling
does and what most eval tooling misses. A conversation can be made of turns that
are each individually correct — grounded, concise, right tool — and still fail:
the customer asks three times, gets three accurate non-answers, and leaves.

On the current traffic: 15 sessions, 39 turns — 7 `unresolved`, 1 `frustrated`,
7 `resolved`. Nearly half of these conversations ended without the customer
getting what they came for, which no turn-level average was going to tell you.

The count of sessions that failed *while every turn in them passed* is **zero**,
and that number is worth more than the one it replaced. An earlier version of
this step scored "every turn passed" against step 03's keyword heuristics alone
and reported two such sessions. Those heuristics flag 10 turns out of 39 where
step 04's evaluators flag 38, so it was clearing sessions the judge had already
condemned. Scored against both signals, as it is now, nothing is clean —
**because the evaluators flag 100% of turns**, which pins the metric at zero
regardless of what happened in any conversation.

So the step prints the zero and then says not to read it as good news: the
measure cannot discriminate until the judge does, and poc/06b is the measurement
showing that same judge agreeing with humans well under half the time. A silent
session is still the failure worth hunting; this fixture just cannot currently
see one.

The verdict is written to the session's *last* span, not to every span in it.
Writing it to all of them would make the mean depend on how many turns each
session happened to take: a chatty session would count five times and a short
one once, and "average session outcome" would silently be an average weighted by
conversation length. Three outcomes rather than pass/fail
(`resolved` / `unresolved` / `frustrated`, scored 1.0 / 0.5 / 0.0), because
"couldn't help" and "made it worse" are different problems with different fixes.

### 02b — spans without OpenTelemetry

The rest of the tour reaches AX through the OTel exporter, which is the right
path for a live application and the wrong one for history: last quarter's
conversations are rows in a warehouse, not processes you can instrument.
`client.spans.log()` takes a dataframe shaped like OpenInference spans and
ingests it directly — no tracer, no collector, no running agent.

It goes to its own `<project>-backfill` project. Mixing backfilled rows into the
project step 03 analyses would change every count that follows — span totals,
failure rates, the dataset built from them — with nothing marking them as a
different source.

Two things worth knowing. HTTP 200 means accepted, not queryable: the spans took
about 90 seconds to appear, so the read-back polls rather than reading once. And
until ingestion has created the project, the export raises
`unauthorized … model does not exist` rather than returning nothing, which reads
as a credentials problem and isn't one. Span ids are derived from the ticket
rather than random, so re-running replaces the history instead of doubling it.

### 03 — the failure classification goes back onto the spans

Step 03 computed which turns failed and wrote it to a local parquet, which meant
the one place you could not ask "show me the hallucinations" was the trace view.
It now writes `metadata.failure_mode`, `failure_count`, `question_id`,
`expected_behavior` and `answer_words` back with `spans.update_metadata()`, so
the classification is a filterable span field.

Metadata rather than `eval.*`: these are code checks over a local fixture, not a
graded judgement, and mixing them into the eval namespace would put two
different kinds of claim in the same place. A clean turn is tagged `"none"`
rather than `""`, because an empty string and an unset field are the same thing
in a UI filter and "no failures" is a finding.

### 07 — the dataset as a living artefact

The human review from step 06 used to stop at the spans. It now reaches the
dataset the experiments actually run against, by two different routes:
`update_examples()` adds `human_label` and `human_reason` as fields on the row,
and `annotate_examples()` records the same verdict as an annotation keyed by
annotation config.

**The field route is verified; the annotation route is not.**
`annotate_examples()` is accepted and raises nothing, but the annotations do not
come back through `list_examples()` — checked repeatedly over several minutes.
Either the write is not landing or the read path does not surface it, and from
the SDK the two are indistinguishable, so the step reports what it can confirm
and points at the UI for the rest. Note also that a categorical annotation
config rejects an explicit `score`: the label determines it, and sending one
422s the whole batch.

Two defects surfaced while building this. Re-running step 07 appended a full
duplicate set every time — `copilot-failures` had reached 105 examples for 35
turns, silently tripling any experiment reading it — so it now appends only
examples whose `source_span_id` is not already present. And appending does *not*
create a new dataset version, contrary to what this README previously claimed;
it writes into the latest one. `--new-version` is what cuts a new one.

### 08 — the model as the second variable

Experiments compared prompts only. The same machinery compares models: a third
arm runs the candidate prompt on `deepseek-v4-flash`, and `--compare-model ''`
turns it off.

Each arm is measured against the run that differs from it by exactly one thing.
The model arm is compared against **v2-on-pro**, not against the v1 baseline —
comparing v2-on-flash to v1-on-pro would change the prompt and the model at once
and could attribute the difference to neither. Both arms go through the same
paired McNemar test, from the same function, because a second comparison written
inline on looser terms is how a cheaper model ends up looking free.

---

## Three steps that need you, not the script

**Step 05 needs an AI integration.** Online LLM judges execute on Arize's
infrastructure, so they need a model provider registered *in the space* — your
local `DEEPSEEK_API_KEY` is not visible to them. Arize has no first-class
DeepSeek provider, so it's registered as a **Custom** OpenAI-compatible one
(base URL + key + model names). Either run
`poc/05_online_evals.py --create-integration` — which uploads your DeepSeek key
to Arize, hence the flag — or add it by hand under **Space Settings →
Integrations → Custom** and set `ARIZE_AI_INTEGRATION_ID`. The code evaluator
works either way.

**Step 07 → Prompt Playground.** Not scriptable, and worth doing by hand: open
the `copilot-failures` dataset in Playground — and while you are there, check
whether the human verdicts step 07 wrote show as annotations on the examples,
which is the one claim in this repo the SDK cannot verify for itself, edit the system prompt against
those exact inputs, and watch the answers change side by side. Step 08 is the
automated version of the same idea.

**Step 10 uses GraphQL, and writes are enterprise-only.** Monitors and dashboards
aren't in the Python SDK. `ARIZE_API_KEY` authenticates against GraphQL with the
`x-api-key` header, so no separate key is needed (`ARIZE_GRAPHQL_API_KEY` still
wins if set), and the endpoint is region-derived like every other host here.
Reads work on any plan — that is how the script's dimension names were verified
against the live schema — but `--apply` returns *"GraphQL Mutation access is only
available for enterprise accounts"*. The script prints the exact settings and
degrades to a "create these by hand" message; the monitor configuration is
identical either way.

An eval score is a **dimension**, not a performance metric: `PerformanceMetric`
is the classic-ML enum (`accuracy`, `auc`, `rmse`, …) and
`CreatePerformanceMonitorMutationInput` has no field for an eval column at all.
Eval monitors are therefore `createDataQualityMonitor` with
`dimensionCategory: llmEval`, `dimensionName: "eval.<name>.score"` and an
aggregation (`avg`). Latency is `latency_ms` under `spanProperty` with `p95`.

---

## Two things steps 09 and 10 surface

**Step 09 promotes v2, and step 08 says v2 isn't proven.** The `production`
label is a pointer, not an endorsement — the step demonstrates that moving it
changes runtime behaviour with no code change, and moving it back is the
rollback. Its verification probe is worth reading: v2, loaded from Prompt Hub,
still answers "no, there are no prorated refunds." The KB documents that
cancellation stops future charges and says *nothing* about proration, so that
answer is inferred from an adjacent policy — the exact move v2's own grounding
section forbids. That is the same finding step 08 measured, in one concrete row.

**Eval column names are case-sensitive, and collisions are invisible.** Columns
are keyed by the evaluator's name verbatim, so an online evaluator named
`Groundedness` (step 05) and step 04's `groundedness` are two unrelated metrics:

```
eval.Groundedness.score   mean 0.33    3 rows   ← online, from step 05
eval.groundedness.score   mean 0.46   39 rows   ← offline, from step 04
```

A monitor can only watch one, and the other regresses unwatched — while the
monitor sits green. Step 10 flags the collision and repoints a monitor whose
metric differs from the logged column only by case.

Also worth trying by hand: **Alyx**, which can author an evaluator from a
natural-language description, and **Agent Studio**.

---

## Layout

```
data/kb/*.md              14 Nimbus support docs — with the deliberate refund gap
data/questions.jsonl      43 questions labelled with expected behaviour
data/orders.json          fake order DB behind lookup_order

src/copilot/
  tracing.py              register() + OpenAIInstrumentor + span helpers
  kb.py                   BM25 retrieval, emits the RETRIEVER span
  tools.py                search_docs / lookup_order / escalate_ticket + v1|v2 schemas
  prompts.py              V1, V2, and the Prompt Hub loader
  evals.py                the groundedness + session judgements, and the aligner
  agent.py                run_turn / run_conversation — AGENT + CHAIN spans, sessions

poc/00..10_*.py           the tour, one script per feature area
poc/02b, 04b, 06b         side paths: backfill, session evals, judge alignment
poc/_common.py            console chrome, client, time windows, the McNemar test
```

Trace shape produced by one turn:

```
AGENT      copilot.turn
├─ CHAIN     router.classify   → LLM   (auto-instrumented)
├─ LLM       answer generation        (auto-instrumented, may repeat)
├─ RETRIEVER kb.search                (retrieval.documents)
├─ TOOL      lookup_order
└─ TOOL      escalate_ticket
```

---

## Notes on the build

- **Model**: `deepseek-v4-pro` for the agent and the judges, `deepseek-v4-flash`
  for the intent router — reached through the `openai` SDK against
  `https://api.deepseek.com/v1`, since the API is OpenAI-protocol-compatible.
  That is also why `OpenAIInstrumentor` produces correct LLM spans for it.
  (`deepseek-chat` / `deepseek-reasoner` were retired on 2026-07-24.)
- **Thinking mode is on by default on V4**, and three things follow from it:
  `thinking` is a DeepSeek extension so it travels in `extra_body`; the router
  explicitly disables it, because on a 32-token budget the CoT would consume
  the whole allowance and return empty content; and inside a tool loop the
  model's `reasoning_content` must be **echoed back** on every subsequent
  request — omitting it is a 400. That last one is the reverse of the
  no-tool-call case, where the CoT is dropped from the context instead.
- Phoenix's judge downgrades from `response_format: json_schema` to tool
  calling on its first call, since DeepSeek rejects the former.
- **Retrieval is BM25, in-process.** A vector DB would add setup friction
  without changing what's being demonstrated — the RETRIEVER span and its
  `retrieval.documents` payload are what AX binds to either way.
- **Manual tool loop, not the SDK tool runner.** Each tool needs its own TOOL
  span and a per-turn trajectory record, which is what the code evaluators grade.
- **Evals are written wide**, as `eval.<name>.{label,score,explanation}` and
  `annotation.<name>.{label,score,text,...}` — that's what the SDK's validators
  actually require. (Phoenix's `to_annotation_dataframe()` emits a long format
  that matches neither; `make check` asserts the column patterns.)
- **Scope**: excludes SSO/RBAC, audit logs, compliance, guardrails, red-teaming,
  CI/CD experiments, and Agent Experiments against a live endpoint (that needs a
  deployed HTTP service, which the CLI-script shape doesn't provide).

## Resetting

Point `ARIZE_PROJECT_NAME` at a fresh project name in `.env` and re-run
`make all` for a clean tour. `rm -rf .out` clears the local intermediates that
scripts pass between each other.

**Do not delete a project you intend to reuse the name of.** Deleting it and
re-running the tour under the same name fails in the worst possible way: the
collector still accepts every span and returns no error, ingestion recreates a
project with that name and a new id, and the spans are never queryable. Both
`spans.list()` and `export_to_df()` return nothing, indefinitely — so step 03
reports "No spans returned" and the whole evaluate stage has nothing to stand
on. Tracing the identical traffic to an unused name works within a minute,
which is how you tell the two apart.

So a reset is a rename, not a delete. The other objects — datasets,
experiments, evaluators, tasks, prompts, annotation configs and queues — do
delete cleanly and can be recreated under their original names.

The one thing worth keeping across resets is the AI integration
(`ARIZE_AI_INTEGRATION_ID`): recreating it means re-uploading your model
provider key, and step 05 finds it by id.

Renaming the project does not rename anything else, so a second tour reuses the
same dataset and evaluator. That is usually what you want — the evaluator keeps
its version history, and step 07 now appends only examples it hasn't seen. Pass
`poc/07_dataset.py --name <something>` if you want a dataset scoped to one tour
instead.

The review queue is the exception, and it has to be: a queue holds spans from
one project, so step 06 names it `Groundedness review (<project>)` and each tour
gets its own. The annotation config stays shared, because that one is a label
schema rather than a pile of records.

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
| 03 | `poc/03_query_spans.py` | Export spans, reconstruct trajectories, **find the failures** | [view traces](https://arize.com/docs/ax/observe/tracing/view-and-manage-traces) |
| 04 | `poc/04_offline_evals.py` | LLM-as-judge + code evals, logged back onto spans | [evals on traces](https://arize.com/docs/ax/evaluate/run-evals-on-traces) |
| 05 | `poc/05_online_evals.py` | Evaluators + **continuous tasks** (sampling, filters) | [online judge](https://arize.com/docs/ax/concepts/evaluators/online-llm-as-judge) · [code evals](https://arize.com/docs/ax/concepts/evaluators/online-code-evaluators) |
| 06 | `poc/06_annotations.py` | Annotation config, **review queue**, judge-vs-human agreement | [human review](https://arize.com/docs/ax/evaluate/human-review) · [align evals](https://arize.com/docs/ax/evaluate/align-evals-to-human-feedback) |
| 07 | `poc/07_dataset.py` | Dataset from failing traces (+ a control group) | [build a dataset](https://arize.com/docs/ax/improve/build-a-dataset) |
| 08 | `poc/08_experiments.py` | **v1 vs v2 on identical inputs** — the payoff | [experiments](https://arize.com/docs/ax/improve/set-up-an-experiment) |
| 09 | `poc/09_prompt_hub.py` | Publish, label `production`, load at runtime | [prompt hub](https://arize.com/docs/ax/concepts/prompts/prompt-hub) |
| 10 | `poc/10_monitors.py` | Monitors, alerting, dashboards | [monitoring](https://arize.com/docs/ax/observe/production-monitoring) |

Or in phases:

```bash
make trace      # 01-02
make observe    # 03
make evaluate   # 04-06
make improve    # 07-09
make monitor    # 10
make all        # everything, in order
```

**The acceptance criterion is step 08.** On the current dataset it is met, on
one of three evaluators. Measured on 35 examples against `deepseek-v4-pro`:

| evaluator | v1 | v2 | delta | rows changed | McNemar |
|---|---|---|---|---|---|
| `answers_from_context` | 1.00 | 1.00 | +0.00 ≈ | 0↑ 0↓ | p=1.000 |
| `conciseness` | 0.80 | 0.91 | +0.11 ≈ | 4↑ 0↓ | p=0.125 |
| `groundedness` | 0.43 | 0.63 | **+0.20 ▲** | 7↑ 0↓ | **p=0.016** |

Seven rows fixed and none broken is a real effect, and the paired test says so.
Conciseness moved the same direction with 4↑ 0↓ and still can't be called —
roughly 5–6 unanimous flips are needed for p<0.05 at this n, which is exactly
the kind of "obvious" improvement that isn't yet evidence.

Three tours have now produced groundedness 0.48→0.52, 0.55→0.52, and
0.43→0.63. Those are *not* three runs of one experiment: each tour regenerates
traffic in step 01 and rebuilds the dataset in step 07, so the inputs differ
every time. The swing across them is wider than any single delta, which is why
step 08 tests significance (paired McNemar, since both variants answer the same
inputs) rather than trusting a delta. An earlier version of this script counted
any positive delta as a win, and duly declared victory on noise.

**Two caveats that the tour itself hands you, and that a green checkmark would
otherwise bury.**

*The metric that improved is one step 06 says not to trust yet.* Judge-vs-human
agreement came out at 35–50%, almost all of it the judge flagging answers the
human didn't. Step 08 shows v2 scoring better under that judge; it does not
show v2 hallucinating less. Aligning the judge is the prerequisite, and step 06
prints exactly that warning.

*Two judges disagree wildly on the same spans.* Step 04's offline Phoenix judge
scores the 38 turns at mean 0.47; the AX-hosted online evaluator from step 05
scores the very same spans at 0.11 — visible side by side in step 10's metrics
table, since the case difference keeps them in separate columns. Same model,
same spans, different template and harness. Whichever you wire a monitor to is
the one that defines "groundedness" for your alerts.

The dataset is also selected *on failures* (23 failing turns + 12 controls), so
it deliberately over-samples the cases v2 targets. That is the right shape for
detecting a fix and the wrong shape for estimating production groundedness.

To harden the result: widen the dataset (more traffic in step 01 → more graded
failures in step 04 → more rows in step 07, which raises the detectable effect
size), and align the judge against human labels before treating its score as
the thing you are optimising.

Step 07 builds its dataset from step 04's **eval verdicts**, not step 03's
heuristics. The heuristics are keyword-narrow — `check_ungrounded` only fires on
refund phrasing and found 2 hallucinations in 38 turns, where the judge grading
every answer flagged 20. Run step 04 before step 07 or you get the narrow set.

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
the `copilot-failures` dataset in Playground, edit the system prompt against
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
  agent.py                run_turn / run_conversation — AGENT + CHAIN spans, sessions

poc/00..10_*.py           the tour, one script per feature area
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

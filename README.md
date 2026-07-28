# Arize AX POC — Nimbus Support Copilot

A deliberately imperfect support agent, built to exercise the main features of
[Arize AX](https://arize.com/docs/ax) end to end: **instrument → observe →
evaluate → improve**.

The agent is the instrument, not the point. It is a RAG + tool-use copilot for a
fictional SaaS ("Nimbus"), wired so that every OpenInference span kind appears in
the trace tree, and seeded with four failure modes that AX's evaluators are built
to catch. The tour then fixes those failures and *proves* the fix with an
experiment.

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
| **Wrong tool** | v1's tool descriptions are one-liners (`"Get order info."`), so order questions get routed to `search_docs`. | Code evaluator on the trajectory |
| **Missing escalation** | v1 never mentions escalation, so blocked and angry users never reach a human. | Code evaluator |
| **Verbosity** | v1 says "be thorough" with no length guidance. | Code evaluator |

`v2` fixes all four. Step 08 measures the difference on identical inputs.

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

**The acceptance criterion is step 08.** If v2 doesn't beat v1 on groundedness
and escalation, the loop hasn't demonstrated anything — re-read the per-row
explanations in the Arize experiment view before touching the prompt.

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

**Step 10 needs a GraphQL key.** Monitors and dashboards aren't in the Python
SDK — they're behind the GraphQL API. The script prints the exact mutation and
applies it with `--apply` if `ARIZE_GRAPHQL_API_KEY` is set.

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

.PHONY: help install check trace backfill observe evaluate improve monitor all \
        ls-backfill ls-observe ls-evaluate ls-improve ls-monitor ls-all clean

PY := uv run python

help:
	@echo "Arize AX POC -- the tour, in order"
	@echo ""
	@echo "  make install    Install dependencies (uv)"
	@echo "  make check      00     offline self-check (no credentials needed)"
	@echo ""
	@echo "  make trace      01-02  instrument: generate traced traffic"
	@echo "  make backfill   02b    instrument: log historic spans without OTel"
	@echo "  make observe    03     observe:    export spans, tag and find the failures"
	@echo "  make evaluate   04-06  evaluate:   judges, sessions, alignment, human review"
	@echo "  make improve    07-09  improve:    dataset, experiments, prompt hub"
	@echo "  make monitor    10     monitor:    monitors + dashboard"
	@echo ""
	@echo "  make all        Run the whole tour end to end"
	@echo ""
	@echo "  Lettered steps (02b, 04b, 06b) are side paths off the step they follow."
	@echo ""
	@echo "  COPILOT_IMPL=langgraph make all          Same tour, LangGraph engine, own -lg project"
	@echo "  COPILOT_OBSERVABILITY=both make trace    Same spans to Arize and LangSmith at once"
	@echo "  COPILOT_OBSERVABILITY=langsmith make all Traces to LangSmith only; 02b-10 skip"
	@echo ""
	@echo "  make ls-all     The LangSmith mirror of the tour (steps ls03-ls10);"
	@echo "                  needs COPILOT_OBSERVABILITY=langsmith or both for traffic"

install:
	uv sync

check:
	$(PY) poc/00_selfcheck.py

trace:
	$(PY) poc/01_trace.py
	$(PY) poc/02_customize_traces.py

# Not in `all`: it writes to its own project and is independent of the tour.
backfill:
	$(PY) poc/02b_log_spans.py

observe:
	$(PY) poc/03_query_spans.py

evaluate:
	$(PY) poc/04_offline_evals.py
	$(PY) poc/04b_session_evals.py
	$(PY) poc/05_online_evals.py
	$(PY) poc/06_annotations.py
	$(PY) poc/06b_align_judge.py

improve:
	$(PY) poc/07_dataset.py
	$(PY) poc/08_experiments.py
	$(PY) poc/09_prompt_hub.py

monitor:
	$(PY) poc/10_monitors.py

all: trace observe evaluate improve monitor

# ---- the LangSmith mirror (poc/ls*.py) ------------------------------------
# Same tour against LangSmith's platform APIs. Traffic comes from the shared
# `make trace` run with COPILOT_OBSERVABILITY=langsmith or both.

# Not in `ls-all`: writes to its own project, independent of the tour (like 02b).
ls-backfill:
	$(PY) poc/ls02b_log_runs.py

ls-observe:
	$(PY) poc/ls03_query_runs.py

# ls06 runs before ls05: the routing rule ls05 creates needs ls06's queue.
ls-evaluate:
	$(PY) poc/ls04_offline_evals.py
	$(PY) poc/ls04b_thread_evals.py
	$(PY) poc/ls06_annotations.py
	$(PY) poc/ls05_online_rules.py
	$(PY) poc/ls06b_align_judge.py

ls-improve:
	$(PY) poc/ls07_dataset.py
	$(PY) poc/ls08_experiments.py
	$(PY) poc/ls09_prompt_hub.py

ls-monitor:
	$(PY) poc/ls10_dashboards.py

ls-all: ls-observe ls-evaluate ls-improve ls-monitor

clean:
	rm -rf .venv **/__pycache__ .out

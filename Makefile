.PHONY: help install check trace observe evaluate improve monitor all clean

PY := uv run python

help:
	@echo "Arize AX POC -- the tour, in order"
	@echo ""
	@echo "  make install    Install dependencies (uv)"
	@echo "  make check      00     offline self-check (no credentials needed)"
	@echo ""
	@echo "  make trace      01-02  instrument: generate traced traffic"
	@echo "  make observe    03     observe:    export spans, find the failures"
	@echo "  make evaluate   04-06  evaluate:   offline judge, online tasks, human review"
	@echo "  make improve    07-09  improve:    dataset, experiments, prompt hub"
	@echo "  make monitor    10     monitor:    monitors + dashboard"
	@echo ""
	@echo "  make all        Run the whole tour end to end"

install:
	uv sync

check:
	$(PY) poc/00_selfcheck.py

trace:
	$(PY) poc/01_trace.py
	$(PY) poc/02_customize_traces.py

observe:
	$(PY) poc/03_query_spans.py

evaluate:
	$(PY) poc/04_offline_evals.py
	$(PY) poc/05_online_evals.py
	$(PY) poc/06_annotations.py

improve:
	$(PY) poc/07_dataset.py
	$(PY) poc/08_experiments.py
	$(PY) poc/09_prompt_hub.py

monitor:
	$(PY) poc/10_monitors.py

all: trace observe evaluate improve monitor

clean:
	rm -rf .venv **/__pycache__ .out

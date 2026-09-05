PYTHON ?= .venv/bin/python
RUFF ?= .venv/bin/ruff
PAPER_DIR := paper/iclr2027
PAPER_NAME := relayspec_iclr2027
SCALING_RAW_ROOT ?= .

.PHONY: help install test test-all lint format check paper paper-assets audit-paper clean research-plan submission-ready

help:
	@echo "install       Install the locked development environment"
	@echo "check         Lint, formatting checks, and CPU code/evidence tests"
	@echo "test-all      Every test, including built-manuscript checks"
	@echo "paper         Compile the current manuscript"
	@echo "paper-assets  Regenerate tables and figures"
	@echo "audit-paper   Compile and run the strict submission-artifact audit"
	@echo "research-plan Validate the research plan and show next tasks (no GPU jobs)"
	@echo "submission-ready Require completion evidence for the research execution plan"
	@echo "clean         Remove Python/LaTeX intermediates; keep evidence and PDF"

install:
	uv sync --locked

test:
	$(PYTHON) -m pytest -q -m "not manuscript"

test-all:
	$(PYTHON) -m pytest -q

lint:
	$(RUFF) check src scripts tests
	$(RUFF) format --check src scripts tests

format:
	$(RUFF) format src scripts tests

check: lint test

research-plan:
	$(PYTHON) scripts/check_submission_plan.py

submission-ready:
	$(PYTHON) scripts/check_submission_plan.py --require-complete

paper-assets:
	$(PYTHON) scripts/build_iclr_paper_assets.py --output $(PAPER_DIR)

paper-scaling-assets:
	$(PYTHON) scripts/build_scaling_paper_assets.py --raw-root $(SCALING_RAW_ROOT) --output $(PAPER_DIR)

paper:
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_NAME).tex
	cd $(PAPER_DIR) && bibtex $(PAPER_NAME)
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_NAME).tex
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_NAME).tex

audit-paper: paper
	$(PYTHON) scripts/audit_iclr_manuscript.py

clean:
	$(PYTHON) -c 'from pathlib import Path; import shutil; [shutil.rmtree(p) for top in ("src", "scripts", "tests") for p in Path(top).rglob("__pycache__")]; [p.unlink() for ext in ("aux", "log", "out", "bbl", "blg", "fls", "fdb_latexmk", "synctex.gz") for p in Path("paper/iclr2027").glob("*." + ext)]'

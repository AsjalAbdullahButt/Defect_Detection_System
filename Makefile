# Developer entry points. Recipes only call `uv`, so they behave the same under sh (Linux/CI,
# Git Bash) and cmd.exe. uv is used as installer + pip-compile; lockfiles are requirements/*.txt.

PYTHON_VERSION := 3.12
COMPILE := uv pip compile --universal --python-version $(PYTHON_VERSION) --generate-hashes --quiet
RUN := uv run
DATA_DIR ?= data/raw

.DEFAULT_GOAL := help
.PHONY: help lock setup setup-serve lint format typecheck test inspect

help: ## List available targets
	@uv run --no-project python -c "import re; [print(f'{m[0]:<14} {m[1]}') for m in re.findall(r'^([a-z-]+):.*?## (.*)$$', open('Makefile').read(), re.M)]"

lock: ## Re-resolve hash-pinned lockfiles from requirements/*.in (serve -> train -> dev)
	$(COMPILE) requirements/base.in requirements/serve.in -o requirements/serve.txt
	$(COMPILE) requirements/base.in requirements/train.in -c requirements/serve.txt -o requirements/train.txt
	$(COMPILE) requirements/dev.in -c requirements/train.txt -c requirements/serve.txt -o requirements/dev.txt

setup: ## Create .venv (Python 3.12) with train+serve+dev deps from lockfiles, hashes enforced
	uv venv --python $(PYTHON_VERSION) --allow-existing .venv
	uv pip sync --require-hashes requirements/train.txt requirements/serve.txt requirements/dev.txt
	uv pip install --no-deps -e .

setup-serve: ## Create .venv with serving deps only (mirrors the production image)
	uv venv --python $(PYTHON_VERSION) --allow-existing .venv
	uv pip sync --require-hashes requirements/serve.txt
	uv pip install --no-deps -e .

lint: ## ruff lint + format check + mypy
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy

format: ## Auto-fix lint issues and format code
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

typecheck: ## mypy only
	$(RUN) mypy

test: ## Run the test suite with coverage
	$(RUN) pytest --cov --cov-report=term

inspect: ## Inventory the raw dataset (override with DATA_DIR=...)
	$(RUN) defect-detection inspect $(DATA_DIR) --json-out reports/inventory.json

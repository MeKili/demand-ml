# demand-ml — task shortcuts.
# Windows note: if you don't have `make`, run the command under each target directly.

.PHONY: install lint fmt typecheck test check train serve

install:        ## Install deps (incl. dev) into the uv-managed venv
	uv sync

lint:           ## Ruff lint
	uv run ruff check .

fmt:            ## Ruff format (writes changes)
	uv run ruff format .

typecheck:      ## mypy (strict) on the package
	uv run mypy src

test:           ## Run the test suite
	uv run pytest

check: lint typecheck test   ## Everything CI runs

train:          ## Train the baseline model on real UCI data
	uv run python -m demand_ml.train

serve:          ## Start the FastAPI server (dev mode with auto-reload)
	uv run uvicorn demand_ml.serving:app --reload

# makefile for llama-model-uploader project
# provides convenient commands for development and usage

.PHONY: help install install-dev install-all test lint format type-check clean run

# default target
help:
	@echo "Available commands:"
	@echo "  install      - install the package in development mode"
	@echo "  install-dev  - install with development dependencies"
	@echo "  install-all  - install with all optional dependencies"
	@echo "  test         - run tests"
	@echo "  lint         - run linting with ruff"
	@echo "  format       - format code with black and isort"
	@echo "  type-check   - run type checking with mypy"
	@echo "  clean        - clean up build artifacts and cache"
	@echo "  run          - run the upload tool (requires MODEL_PATH and REPO_NAME)"
	@echo "  lock         - update dependency lock file"

# installation targets
install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

install-all:
	uv pip install -e ".[all,dev]"

# development targets
test:
	uv run pytest

lint:
	uv run ruff check *.py llama_uploader/*.py

format:
	uv run black *.py llama_uploader/*.py
	uv run isort *.py llama_uploader/*.py

type-check:
	uv run mypy llama_uploader/*.py

# utility targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

# run the tool (requires environment variables or parameters)
run:
	@if [ -z "$(MODEL_PATH)" ] || [ -z "$(REPO_NAME)" ]; then \
		echo "Usage: make run MODEL_PATH=./path/to/model REPO_NAME=user/model-name"; \
		echo "Optional: HF_TOKEN=your_token PRIVATE=true TEST=true"; \
		exit 1; \
	fi
	uv run llama-upload \
		--model_path $(MODEL_PATH) \
		--repo_name $(REPO_NAME) \
		$(if $(HF_TOKEN),--hf_token $(HF_TOKEN)) \
		$(if $(PRIVATE),--private) \
		$(if $(TEST),--test)

# dependency management
lock:
	uv lock

# setup development environment from scratch
setup-dev:
	uv venv
	@echo "Virtual environment created. Activate with:"
	@echo "source .venv/bin/activate  # linux/macos"
	@echo ".venv\\Scripts\\activate     # windows"
	@echo "Then run: make install-dev"

# validate installation
validate:
	uv run python -c "import llama_uploader; from llama_uploader import detect_llama_version; print('✅ All modules imported successfully')"
	uv run llama-upload --help

# build distribution
build:
	uv build

# full development workflow
dev-setup: setup-dev install-dev format type-check lint validate
	@echo "✅ Development environment ready!"

# ci workflow
ci: install-dev format type-check lint test
	@echo "✅ CI checks passed!"

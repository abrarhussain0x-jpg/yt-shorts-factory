.PHONY: help venv install install-all run test clean lint worker queue history stats verify config analyze transcribe info cleanup interactive

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

venv: ## Create virtual environment
	python3 -m venv .venv
	@echo "Activate with: source .venv/bin/activate"

install: venv ## Install dependencies and package
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .
	@echo "Installation complete. Activate with: source .venv/bin/activate"

install-all: venv ## Install with all optional dependencies
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e ".[all]"
	@echo "Full installation complete (face tracking, faster whisper, visualization)"

run: ## Run the pipeline (use ARGS="..." for options)
	@.venv/bin/python main.py run $(ARGS)

test: ## Run tests
	.venv/bin/python -m pytest tests/ -v

clean: ## Remove output, caches, and venv
	rm -rf output/ __pycache__ */__pycache__ */*/__pycache__ .pytest_cache .venv
	@echo "Cleaned up"

lint: ## Run linters
	@.venv/bin/python -m py_compile main.py
	@.venv/bin/python -m py_compile core/*.py
	@.venv/bin/python -m py_compile config/*.py
	@.venv/bin/python -m py_compile utils/*.py
	@.venv/bin/python -m py_compile database/*.py
	@.venv/bin/python -m py_compile scheduler/*.py
	@echo "Syntax check passed"

worker: ## Start the worker daemon
	@.venv/bin/python main.py worker --start $(ARGS)

queue: ## Queue a URL (use ARGS="--url ...")
	@.venv/bin/python main.py queue $(ARGS)

history: ## Show job history
	@.venv/bin/python main.py history $(ARGS)

stats: ## Show pipeline statistics
	@.venv/bin/python main.py stats $(ARGS)

verify: ## Verify all dependencies
	@.venv/bin/python main.py verify

config: ## Show current configuration
	@.venv/bin/python main.py config $(ARGS)

analyze: ## Analyze video without full pipeline
	@.venv/bin/python main.py analyze $(ARGS)

transcribe: ## Transcribe a local video file
	@.venv/bin/python main.py transcribe $(ARGS)

info: ## Show info about a video
	@.venv/bin/python main.py info $(ARGS)

cleanup: ## Clean up old files (use ARGS="--days 30 --dry-run")
	@.venv/bin/python main.py cleanup $(ARGS)

interactive: ## Interactive mode with guided workflow
	@.venv/bin/python main.py interactive

setup: ## One-shot setup (install all deps + download whisper model)
	bash scripts/install_deps.sh

zip: ## Create distribution zip
	@cd .. && zip -r yt-shorts-factory.zip yt-shorts-factory/ -x "yt-shorts-factory/.venv/*" "yt-shorts-factory/output/*" "yt-shorts-factory/__pycache__/*" "yt-shorts-factory/*/__pycache__/*" "yt-shorts-factory/*/*/__pycache__/*"
	@echo "Created ../yt-shorts-factory.zip"

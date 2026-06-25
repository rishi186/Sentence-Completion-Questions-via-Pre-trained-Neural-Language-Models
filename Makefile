# ESL Sentence Completion - Makefile

.PHONY: help install dev-install test test-cov lint format typecheck run app pdf clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

dev-install: ## Install dev dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"

test: ## Run tests
	python -m pytest tests/ -v

test-cov: ## Run tests with coverage
	python -m pytest tests/ --cov=src --cov-report=term-missing

lint: ## Lint code
	ruff check src/ tests/ main.py app.py

format: ## Format code
	black src/ tests/ main.py app.py

typecheck: ## Type check
	mypy src/

app: ## Run Streamlit app
	streamlit run app.py

pdf: ## Generate sample PDF
	python generate_sample_pdf.py

clean: ## Clean build artifacts
	rm -rf __pycache__ .pytest_cache *.egg-info dist build
	find . -type d -name __pycache__ -exec rm -rf {} +

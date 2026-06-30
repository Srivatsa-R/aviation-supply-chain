# ============================================================
# Aviation Supply Chain Analytics — Makefile
# ============================================================
# Usage:
#   make install    — Set up the Python environment
#   make data       — Download and clean all datasets
#   make test       — Run all unit tests
#   make lint       — Check code quality
#   make run        — Launch the Streamlit dashboard
#   make notebooks  — Execute and export all notebooks to HTML
#   make clean      — Remove generated files
# ============================================================

.PHONY: install data test lint run notebooks clean help

# Default target
help:
	@echo ""
	@echo "Aviation Supply Chain Analytics — Available Commands"
	@echo "====================================================="
	@echo "  make install    Install Python dependencies"
	@echo "  make data       Download & clean all datasets"
	@echo "  make test       Run unit tests (pytest)"
	@echo "  make lint       Run Black + isort + flake8"
	@echo "  make run        Launch Streamlit dashboard"
	@echo "  make notebooks  Execute all notebooks → HTML"
	@echo "  make clean      Remove generated files"
	@echo ""

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

data:
	python -c "from src.data.ingest import load_all_datasets; load_all_datasets()"
	@echo "✅ Datasets ready"

test:
	pytest tests/ -v --tb=short

lint:
	black src/ tests/
	isort src/ tests/
	flake8 src/ tests/ --max-line-length=100

run:
	streamlit run app/streamlit_app.py

notebooks:
	@for nb in notebooks/*.ipynb; do \
		echo "Running $$nb..."; \
		jupyter nbconvert --to html --execute --output-dir docs/ "$$nb"; \
	done
	@echo "✅ Notebooks exported to docs/"

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -delete
	rm -f supply_chain_network.html
	@echo "✅ Cleaned up generated files"

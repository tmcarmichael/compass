# compass development commands

set dotenv-load := false

# List available commands
default:
    @just --list

# Run the full test suite
test *args='':
    python3 -m pytest tests/ -v {{args}}

# Run tests with hypothesis CI profile (more examples)
test-ci:
    python3 -m pytest tests/ -v --hypothesis-profile=ci --cov --cov-report=term-missing --benchmark-disable

# Lint with ruff
lint:
    python3 -m ruff check src/ tests/

# Format check (no writes)
fmt-check:
    python3 -m ruff format --check src/ tests/

# Auto-format
fmt:
    python3 -m ruff format src/ tests/

# Type check
typecheck:
    python3 -m mypy src/

# All checks (mirrors CI)
check: lint fmt-check typecheck test-ci

# Collect tests without running
test-collect:
    python3 -m pytest tests/ --co -q

# Run benchmarks
benchmark:
    python3 -m pytest tests/test_performance.py --benchmark-only -v

# Run headless simulator (replay | benchmark | converge)
simulate *args='':
    PYTHONPATH=src python3 -m simulator {{args}}

# Install pre-commit hooks
install-hooks:
    pre-commit install

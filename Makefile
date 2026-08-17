.PHONY: help install lint test test-unit test-integration data build up down logs clean e2e

help:
	@echo "FedShield targets:"
	@echo "  install           Install Python dependencies"
	@echo "  lint              Run ruff linter"
	@echo "  test              Run unit + integration tests"
	@echo "  data              Generate Non-IID synthetic partitions"
	@echo "  build             Build docker images"
	@echo "  up                Start full stack (docker compose)"
	@echo "  down              Stop stack"
	@echo "  logs              Tail compose logs"
	@echo "  e2e               data + build + up (end-to-end)"
	@echo "  clean             Remove caches and generated artifacts"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	ruff check src tests scripts

test-unit:
	pytest tests/unit -q

test-integration:
	FEDSHIELD_DEV_CERTS=true pytest tests/integration -q

test: test-unit test-integration

data:
	python scripts/generate_non_iid_data.py

build:
	docker compose build

up: data
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=200

e2e: data build
	docker compose up --abort-on-container-exit --exit-code-from server || true
	docker compose ps
	@echo "MLflow UI: http://localhost:5000"

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache htmlcov mlruns
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

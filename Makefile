.PHONY: help install lint test test-unit test-integration data init-env build up down logs clean e2e

help:
	@echo "FedShield targets:"
	@echo "  init-env          Create gitignored .env with a random MLflow password"
	@echo "  install           Install Python dependencies"
	@echo "  lint              Run ruff linter"
	@echo "  test              Run unit + integration tests"
	@echo "  data              Generate Non-IID synthetic partitions (local pytest)"
	@echo "  build             Build docker images"
	@echo "  up                Create .env if needed and start the stack"
	@echo "  down              Stop stack"
	@echo "  logs              Tail compose logs"
	@echo "  e2e               init-env + build + foreground compose"
	@echo "  clean             Remove caches and generated artifacts"

init-env:
	python scripts/init_env.py

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

up: init-env
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=200

e2e: init-env build
	docker compose up --abort-on-container-exit --exit-code-from server

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache htmlcov mlruns
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

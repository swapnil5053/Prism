.PHONY: install lint fmt typecheck test test-unit test-web migrate api worker web eval-oracle up up-gpu

install:
	uv sync
	cd web && npm ci

lint:
	uv run ruff check .
	uv run ruff format --check .
	cd web && npm run lint

fmt:
	uv run ruff check --fix .
	uv run ruff format .
	cd web && npm run format

typecheck:
	uv run mypy

test-unit:
	uv run pytest -m "not integration and not model"

test:
	uv run pytest -m "not model" --cov --cov-report=term-missing

test-web:
	cd web && npm test

migrate:
	uv run alembic upgrade head

api:
	uv run prism-api

worker:
	uv sync --extra worker
	uv run prism-worker

web:
	cd web && npm run dev

eval-oracle:
	uv run prism-eval oracle --data eval/data/synth-test --results eval/results/oracle-test.json

COMPOSE = docker compose --env-file .env -f deploy/compose.yaml

up:
	$(COMPOSE) up --build

up-gpu:
	$(COMPOSE) -f deploy/compose.gpu.yaml up --build

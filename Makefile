.PHONY: install lint fmt typecheck test test-unit test-web migrate api worker web eval-oracle

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

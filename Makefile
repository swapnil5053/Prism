.PHONY: install lint fmt typecheck test test-unit migrate api

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test-unit:
	uv run pytest -m "not integration"

test:
	uv run pytest --cov --cov-report=term-missing

migrate:
	uv run alembic upgrade head

api:
	uv run prism-api

# Prism

Upload a UI screenshot, get back the detected elements and a list of likely
accessibility problems (low text contrast, small touch targets, unlabeled inputs).

Work in progress: the API and storage layer are done; the GPU worker is next.

## Development

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/), Postgres 15+ and Redis 7+.

```bash
cp .env.example .env   # then set PRISM_SECRET_KEY and the database URL
make install
make migrate
make api               # http://127.0.0.1:8000/docs
```

### Tests

```bash
make test-unit         # no services needed
make test              # also runs integration tests
```

Integration tests use `PRISM_TEST_DATABASE_URL` (default
`postgresql+asyncpg://prism:prism-test@localhost/prism_test`) and Redis db 15
(`PRISM_TEST_REDIS_URL`). The test database is wiped on every run.

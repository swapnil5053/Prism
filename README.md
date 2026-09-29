# Prism

Upload a screenshot of a UI and get back the elements on it plus a list of
likely accessibility problems: low text contrast, touch targets under 24 CSS
px, and form controls with no visible label.

A vision-language model (Qwen2.5-VL, 4-bit) finds the elements. The checks
themselves are ordinary code that measures pixels, because a 3B model can't
reliably compute a contrast ratio, and a unit test can check code that does.

## How it works

```
browser ──upload──▶ FastAPI ──enqueue──▶ Redis ──▶ GPU worker (arq)
   ▲                  │  ▲                  │          │ detect (Qwen2.5-VL)
   └──── progress ────┘  └── event stream ──┘          │ audit (pixels)
                      │                                 ▼
                      └──────────── Postgres ◀──── result (JSONB)
```

- **API** (`src/prism/api`): uploads, results, reports, cancel. Anonymous
  workspaces via an HMAC-signed cookie, so one visitor can't see another's
  analyses. Uploads are decoded and re-encoded before storage; the client's
  filename and content type are never trusted.
- **Worker** (`src/prism/worker`): one job at a time on one GPU. Inference runs
  in a thread; a cancel request flips a flag the model checks every token.
- **Events** (`src/prism/events.py`): progress goes into a Redis stream per
  analysis, so a browser that reconnects replays what it missed.
- **Checks** (`src/prism/a11y`): WCAG 2.2 SC 1.4.3 (contrast), 2.5.8 (target
  size), 3.3.2 (visible labels). See [docs/decisions.md](docs/decisions.md)
  for how each one works and where it falls short.
- **Web** (`web/`): plain HTML, CSS and JavaScript, built with Vite.

## Results

Measured on 300 synthetic pages with exact labels, held out from the pages
used while developing the rules ([details](eval/README.md)):

| Check (on true boxes) | Precision | Recall |
|---|---|---|
| Text contrast | 0.870 | 0.931 |
| Target size | 1.000 | 1.000 |
| Visible label | 1.000 | 0.748 |

Contrast estimates are within 4.1% of the true ratio at the median, and the
pass/fail call at 4.5:1 agrees with the truth 99.5% of the time outside a ±10%
band around the threshold.

## Running it

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 20+, Postgres and
Redis. The worker needs an NVIDIA GPU; it's developed on an 8 GB RTX 4060.

```bash
cp .env.example .env        # set PRISM_SECRET_KEY and the database URL
make install migrate
make api                    # http://127.0.0.1:8000/docs
make worker                 # downloads the model on first run
make web                    # http://localhost:5173
```

Or with Docker (set `POSTGRES_PASSWORD` and `PRISM_SECRET_KEY` in `.env`):

```bash
docker compose up --build                                   # API + web, no worker
docker compose -f compose.yaml -f compose.gpu.yaml up --build   # with the GPU worker
```

## Tests

```bash
make test        # unit + integration (needs Postgres and Redis, see below)
make test-web
make lint typecheck
```

Integration tests use `PRISM_TEST_DATABASE_URL` (default
`postgresql+asyncpg://prism:prism-test@localhost/prism_test`) and Redis db 15;
both are wiped on every run. Tests marked `model` run the real detector code
on a tiny randomly initialised checkpoint, so they need the `worker` extra but
no GPU.

## Limitations

- Everything is inferred from pixels. There's no DOM, so no alt text, focus
  order, ARIA or keyboard checks.
- "Large text" for contrast is guessed from box height; bold 14pt text is
  treated as normal text.
- The label check is layout-based: a heading right above an unlabeled input
  looks like its label.
- The detection benchmark on real screenshots hasn't been run yet.

## License

MIT

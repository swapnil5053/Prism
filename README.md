# Prism

Upload a UI screenshot, get back the detected elements and a list of likely
accessibility problems (low text contrast, small touch targets, unlabeled inputs).

Work in progress.

## Development

```bash
cp .env.example .env   # then set PRISM_SECRET_KEY
make install
make lint test-unit
```

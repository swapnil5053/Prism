# PyTorch wheels bundle the CUDA runtime, so a slim base works as long as the
# host has an NVIDIA driver and the container toolkit.
FROM python:3.12-slim AS deps
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra worker --no-install-project
COPY src/ src/
RUN uv sync --frozen --no-dev --extra worker

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 prism
WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
COPY --from=deps /app/src /app/src
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    PRISM_UPLOAD_DIR=/data/uploads
RUN mkdir -p /data/uploads /models && chown -R prism /data /models
USER prism
CMD ["prism-worker"]

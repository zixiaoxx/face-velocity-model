FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
COPY face_velocity ./face_velocity
COPY evaluation ./evaluation
COPY scripts ./scripts
COPY tests ./tests
COPY config ./config
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "face_velocity.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

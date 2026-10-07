FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/uv
COPY pyproject.toml uv.lock ./
COPY medical_agent ./medical_agent
COPY data ./data
COPY web ./web
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "medical_agent.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

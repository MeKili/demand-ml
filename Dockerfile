FROM python:3.12-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src src
COPY models models

# Install uv and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["uv", "run", "uvicorn", "demand_ml.serving:app", "--host", "0.0.0.0", "--port", "8000"]

# syntax=docker/dockerfile:1
FROM python:3.13-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install uv and use the existing lockfile so dependency versions match local dev.
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache --compile .

# Copy the rest of the project (model artifact included, feel free to swap for a bind mount if you prefer).
COPY . /app

EXPOSE 5000
ENV MODEL_PATH=/app/artifacts/model.joblib

CMD ["python", "serve.py"]

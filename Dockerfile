# Build image for Template-Backend (FastAPI + Celery)

FROM python:3.11-slim AS base

LABEL maintainer="Template Backend Team"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies first (leverages docker cache)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Ensure logs directory exists
RUN mkdir -p /app/logs

# Expose API port
EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--log-level", "info", "-b", "0.0.0.0:8000", "main:app"] 
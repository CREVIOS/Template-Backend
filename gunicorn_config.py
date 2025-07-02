"""Gunicorn configuration for FastAPI server
This file configures Gunicorn for production deployment of the FastAPI application.
It uses worker processes with async workers to handle concurrent requests efficiently.
"""

import multiprocessing
import os
from pathlib import Path

# Base directory for the application
BASE_DIR = Path(__file__).parent.resolve()

# The socket to bind
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")

# Number of worker processes
# A good rule of thumb is 2-4 x $(NUM_CORES)
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))

# Worker class - using uvicorn for ASGI support
worker_class = "uvicorn.workers.UvicornWorker"

# Log level
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# Access log file
accesslog = os.getenv("GUNICORN_ACCESS_LOG", str(BASE_DIR / "logs/access.log"))

# Error log file
errorlog = os.getenv("GUNICORN_ERROR_LOG", str(BASE_DIR / "logs/error.log"))

# Timeout in seconds
timeout = int(os.getenv("GUNICORN_TIMEOUT", "600"))  # 10 minutes for AI operations

# Graceful timeout in seconds (time to wait for workers to finish before killing)
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "300"))  # 5 minutes

# Keep alive timeout
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

# Maximum number of simultaneous clients
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))

# Maximum number of requests a worker will process before restarting
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "50"))

# Process name
proc_name = "legal_template_api"

# The maximum number of pending connections
backlog = int(os.getenv("GUNICORN_BACKLOG", "2048"))

# Preload application code before worker processes are forked
preload_app = True

# Restart workers when code changes (development only)
reload = os.getenv("GUNICORN_RELOAD", "false").lower() in ("true", "1", "t")

# SSL certificates for HTTPS (uncomment for production)
# certfile = os.getenv("GUNICORN_CERTFILE", "")
# keyfile = os.getenv("GUNICORN_KEYFILE", "")

# Configure logging
logconfig_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "file": {
            "level": "INFO",
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": str(BASE_DIR / "logs/gunicorn.log"),
            "when": "midnight",
            "backupCount": 7,
            "formatter": "standard",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True,
        },
    },
}

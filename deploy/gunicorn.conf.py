import multiprocessing
import os

# Bind only to loopback; Nginx will reverse proxy
bind = os.getenv("GUNICORN_BIND", "127.0.0.1:8000")

# Sensible default: (2 x CPU) + 1
workers = int(os.getenv("GUNICORN_WORKERS", max(2, multiprocessing.cpu_count() * 2 + 1)))

# Uvicorn worker for ASGI/FastAPI
worker_class = "uvicorn.workers.UvicornWorker"

# Timeouts (tune if endpoints can run long LLM/IO tasks)
timeout = int(os.getenv("GUNICORN_TIMEOUT", "60"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

# Logging to stdout/stderr; systemd/journalctl will capture
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")
accesslog = "-"
errorlog = "-"

# Graceful behavior
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))

# Max requests to mitigate memory leaks (optional)
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "0"))  # 0 disables
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "0"))

# Set forwarded-allow-ips if behind reverse proxy (nginx)
forwarded_allow_ips = "*"

# Docker Deployment Instructions

## Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux) must be installed **and running**.

## Setup

1. **Environment Variables**:
   Ensure your `.env` file in the backend directory has the correct database connection string for Docker:
   ```env
   DATABASE_URL=postgresql+psycopg://tradvio_user:change-me@db:5432/tradvio
   POSTGRES_CONN_HOST=db
   ```

2. **Start Services**:
   Run the following command in this directory:
   ```bash
   docker compose up -d --build
   ```

## Troubleshooting "System cannot find the file specified"
If you see an error like `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`, it means **Docker Desktop is not running**.

1. Open **Docker Desktop** application on your computer.
2. Wait for the engine to start (the whale icon in the taskbar should stop animating).
3. Try the command again.

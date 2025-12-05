# Tradvio Clone Backend (FastAPI)

FastAPI service with PostgreSQL. Tables are created automatically at startup using SQLAlchemy (no Alembic). It can run locally or via Docker Compose.

This project now saves uploaded Analyse images as base64 data URLs in the database (column: `chart_image_data`). Database tables are created automatically at application startup (no Alembic).

## Quick Start (Docker Compose - Development)

Prerequisites:
- Docker and Docker Compose v2

Steps:
1. Copy environment file and adjust values as needed:
   ```bash
   cd tradvio-clone-backend
   cp .env.example .env
   ```
   Key variables to check:
   - `DATABASE_URL=postgresql://tradvio_user:change-me@db:5432/tradvio`
   - `CORS_ORIGINS=http://localhost:5173`
   - Optional AI keys (GEMINI_API_KEY / XAI_API_KEY) if you use the AI endpoints.

2. Start services:
   ```bash
   docker compose up -d --build
   ```
   - This starts Postgres and the API.
   - On container start, the API starts. SQLAlchemy creates tables and seeds default rows if needed.

3. Verify:
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs


## Environment Variables

Create `.env` from `.env.example`. Common ones:
- Core:
  - `DATABASE_URL` (for in-docker Postgres use the provided default)
  - `CORS_ORIGINS` (comma separated; add your frontend origin)
- AI Providers (optional): `GEMINI_API_KEY`, `XAI_API_KEY`
- Fundamentals (optional): `MARKETAUX_API_KEY` / `MARKETEAUX_API_KEY`, `TRADING_ECONOMICS_API_KEY`
- Auth & Cookies: `AUTH_COOKIE_SECURE`, `AUTH_COOKIE_SAMESITE`
- Stripe/Paypal (optional): `STRIPE_API_KEY`, `STRIPE_WEBHOOK_SECRET`, `PAYPAL_*`

## Database Initialization

- On startup, SQLAlchemy imports all models and runs `Base.metadata.create_all(bind=engine)`.
- The app ensures Postgres extension `pgcrypto` is available for `gen_random_uuid()`.
- Default rows are seeded idempotently:
  - A default admin user is ensured to exist and remain active.
  - AI configuration (`AIConfig`) is created if missing with sensible defaults.
- No Alembic migrations are used.

## Production Deployment (Simple Plan)

The application will be hosted on the client’s private server.

### 1) One-time Server Setup
- Provision a Linux VM (Ubuntu 22.04+ recommended)
- Install Docker and Docker Compose v2
- Create a non-root user with docker permissions
- Point DNS of your domain/subdomain to this server (e.g., `api.example.com` A record -> server IP)

### 2) Backend Deployment (Docker Compose)
- SSH into the server and clone the repo:
  ```bash
  git clone <repo> && cd tradvio-clone-backend
  ```
- Create the production `.env` from `.env.example` and set production values:
  - `DATABASE_URL=postgresql://tradvio_user:CHANGE_ME@db:5432/tradvio`
  - `CORS_ORIGINS=https://app.example.com`
  - AI keys and any provider credentials
  - Consider `AUTH_COOKIE_SECURE=true` and appropriate `AUTH_COOKIE_SAMESITE` for HTTPS
- Start services:
  ```bash
  docker compose up -d --build
  ```
- To monitor startup and table creation, run:
  ```bash
  docker compose logs -f api
  ```

## Troubleshooting
- 500 on start: check database connectivity and that tables were created (inspect container logs)
- CORS errors: add frontend origin to `CORS_ORIGINS`
- SSL issues: check DNS and reverse proxy configuration

## Endpoints
- API root: `GET /api/`
- Swagger: `GET /docs`

## Notes
- Uploaded analysis images are stored as base64 in `chart_image_data` and returned in list/details endpoints. Frontend prefers base64 and falls back gracefully.

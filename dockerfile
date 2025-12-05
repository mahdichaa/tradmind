FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libtesseract-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the whole project (main.py at repo root)
COPY . .

# make absolute imports like "from app..." work
ENV PYTHONPATH=/app

EXPOSE 8000

# default CMD runs uvicorn; docker-compose starts the API directly (no Alembic)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

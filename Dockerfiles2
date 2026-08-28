# ─── Base Image ────────────────────────────────────────────────────────────────
# Using Python 3.9-slim for compatibility with TFX 1.11.0 / TF 2.10.1
FROM python:3.9-slim

# ─── Environment Variables ─────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    MODEL_DIR=/app/serving_model_dir/cc-fraud-model

# ─── System Dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ─── Working Directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ─── Install Python Dependencies ──────────────────────────────────────────────
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# ─── Copy Application Files ────────────────────────────────────────────────────
COPY app/ ./app/
COPY serving_model_dir/ ./serving_model_dir/

# ─── Expose Port ───────────────────────────────────────────────────────────────
EXPOSE 5000

# ─── Run Application ──────────────────────────────────────────────────────────
CMD ["python", "app/main.py"]

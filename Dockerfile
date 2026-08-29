FROM python:3.10-slim

# Mencegah Python menulis file .pyc dan buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependensi sistem yang minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Salin requirements dan install terlebih dahulu (agar ter-cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh file proyek
COPY . .

# Expose port (sesuaikan dengan framework Anda, misal FastAPI 8000 / Streamlit 8501 / Flask 5000)
EXPOSE 8000

CMD ["gunicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

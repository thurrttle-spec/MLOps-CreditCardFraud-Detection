FROM python:3.10-slim

# Mencegah Python menulis file .pyc dan buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependensi sistem minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Salin requirements dan install dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh isi repositori ke /app
COPY . .

EXPOSE 8000

# Jalankan Gunicorn langsung di direktori /app tempat main.py berada
CMD ["gunicorn", "-b", "0.0.0.0:8000", "main:app"]

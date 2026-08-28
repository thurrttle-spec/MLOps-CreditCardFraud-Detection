# Use a slim base image
FROM python:3.9-slim

WORKDIR /app

# Install only essentials
RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

EXPOSE 8080

# Run Flask app (adjust if FastAPI)
CMD ["python", "app.py"]



# Use Python 3.11 slim as base
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    UPLOAD_FOLDER=/app/uploads \
    CHROMA_DB_PATH=/app/chroma_db

# Create necessary directories
WORKDIR /app
RUN mkdir -p $UPLOAD_FOLDER $CHROMA_DB_PATH

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -i https://pypi.org/simple -r requirements.txt


# Copy application code
COPY . .

# Set up Firebase Admin SDK credentials
RUN mkdir -p /app/credentials
COPY firebase_key.json /app/credentials/firebase_key.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/firebase_key.json

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run the application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]
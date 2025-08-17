FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    UPLOAD_FOLDER=/app/uploads \
    CHROMA_DB_PATH=/app/chroma_db \
    GOOGLE_APPLICATION_CREDENTIALS=/mnt/secrets/firebase_key.json

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run the app
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]

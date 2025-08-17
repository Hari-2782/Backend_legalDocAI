# Stage 1: Base build stage
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime AS base

WORKDIR /app

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements for caching
COPY requirements.txt .

# Install runtime essentials
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Optional large packages (cached separately)
FROM base AS large-deps

# Install large optional packages
RUN pip install --no-cache-dir torchvision sentence-transformers

# Stage 3: Runtime stage (minimal)
FROM base AS runtime

WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    UPLOAD_FOLDER=/app/uploads \
    CHROMA_DB_PATH=/app/chroma_db \
    GOOGLE_APPLICATION_CREDENTIALS=/mnt/secrets/firebase_key.json

# Copy installed dependencies from base
COPY --from=base /usr/local/lib/python*/site-packages /usr/local/lib/python*/site-packages
COPY --from=base /app /app

# Copy large packages (cached)
COPY --from=large-deps /usr/local/lib/python*/site-packages/torchvision* /usr/local/lib/python*/site-packages/
COPY --from=large-deps /usr/local/lib/python*/site-packages/sentence_transformers* /usr/local/lib/python*/site-packages/

# Remove unnecessary files
RUN rm -rf /app/tests /app/docs /root/.cache

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run the app
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]

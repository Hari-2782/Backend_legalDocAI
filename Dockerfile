# ---- Builder Stage ----
    FROM python:3.11-slim AS builder

    WORKDIR /opt/build
    
    # Install system dependencies required for building python packages
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpoppler-cpp-dev \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy only the requirements file
    COPY requirements.txt .
    
    # Create a filtered requirements file (exclude only heavy training libs)
    RUN grep -v -E "^(transformers|datasets|peft|accelerate)" requirements.txt > runtime-requirements.txt
    
    # Create and activate a virtual environment
    RUN python -m venv /opt/venv
    ENV PATH="/opt/venv/bin:$PATH"
    
    # Install torch (CPU-only) first, then runtime dependencies
    RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
    RUN pip install --no-cache-dir -r runtime-requirements.txt
    
    
    # ---- Final Stage ----
    FROM python:3.11-slim
    
    WORKDIR /app
    
    # Install only minimal system dependency for PyMuPDF
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libpoppler-glib8 \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy the virtual environment from builder
    COPY --from=builder /opt/venv /opt/venv
    
    # Copy application code
    COPY app/ ./app
    
    # Env setup
    ENV PATH="/opt/venv/bin:$PATH"
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    ENV PORT=8080
    
    EXPOSE 8080
    
    # Run app
    CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]
    
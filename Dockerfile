# ─────────────────────────────────────────────────────────────────────────────
# Cat Rescue OpenEnv — Dockerfile
# Compatible with Hugging Face Spaces (port 7860 required)
#
# Build:  docker build -t cat-rescue .
# Run:    docker run -p 7860:7860 cat-rescue
# ─────────────────────────────────────────────────────────────────────────────

# Use the official slim Python 3.12 image as the base
FROM python:3.12-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="Cat Rescue Team"
LABEL description="Cat Rescue OpenEnv — Meta × PyTorch Hackathon"

# ── System setup ──────────────────────────────────────────────────────────────
# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
# (unbuffered output is important for real-time logs in HF Spaces)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Install dependencies ───────────────────────────────────────────────────────
# Copy only requirements first so Docker can cache this layer
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────────────
# Copy all project files into the container's working directory
COPY environment.py .
COPY rewards.py     .
COPY grader.py      .
COPY server.py      .

# ── Expose the port ───────────────────────────────────────────────────────────
# HuggingFace Spaces routes external traffic to port 7860
EXPOSE 7860

# ── Launch command ─────────────────────────────────────────────────────────────
# Start the FastAPI app via uvicorn on 0.0.0.0:7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]

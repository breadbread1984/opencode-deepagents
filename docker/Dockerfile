FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create workspace and data directories
RUN mkdir -p /workspace /root/.opencode-deepagents

ENV DEFAULT_WORKSPACE=/workspace
ENV CHECKPOINT_DB=/root/.opencode-deepagents/checkpoints.db
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

ENTRYPOINT ["python", "app.py"]
CMD ["--host", "0.0.0.0", "--port", "7860"]

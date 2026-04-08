
# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install system deps (lightweight)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip

# 🔥 IMPORTANT: install required libs
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose HF port
EXPOSE 7860

# Start server
CMD ["python", "-m", "server.app"]
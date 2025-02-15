FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x and npm
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get update && apt-get install -y nodejs && \
    npm --version && \
    node --version

# Set working directory
WORKDIR /app

# Create data directory with proper permissions
RUN mkdir -p /data && chown -R 1000:1000 /data && chmod 777 /data

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install prettier globally with specific version
RUN npm install -g prettier@3.4.2 && \
    mkdir -p /app/node_modules && \
    chmod -R 777 /app/node_modules && \
    prettier --version

# Copy application code
COPY . .

# Set proper permissions for the app directory
RUN chmod -R 755 /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/.npm && \
    chmod -R 777 /app/.npm

# Expose port
EXPOSE 8000

# Environment variables
ENV NODE_PATH=/usr/lib/node_modules
ENV PATH="/app/node_modules/.bin:${PATH}"
ENV PYTHONPATH="/app"

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
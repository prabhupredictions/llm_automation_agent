#!/bin/bash

# Check if AIPROXY_TOKEN is provided
if [ -z "$AIPROXY_TOKEN" ]; then
    echo "Error: AIPROXY_TOKEN is required"
    exit 1
fi

# Create data directory if not exists
echo "Setting up test environment..."
mkdir -p "$(pwd)/data"
chmod 777 "$(pwd)/data"  # Ensure directory is writable

# Remove existing container if it exists
docker rm -f llm-agent 2>/dev/null

# Build the Docker image
echo "Building Docker image..."
docker build -t llm-automation-agent .

# Run the container
echo "Starting container..."
docker run -d \
    --name llm-agent \
    -p 8000:8000 \
    -e AIPROXY_TOKEN=$AIPROXY_TOKEN \
    -v "$(pwd)/data:/data" \
    llm-automation-agent

# Wait for the application to start
echo "Waiting for application to start..."
sleep 5

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run tests
echo "Running tests..."
PYTHONPATH=. pytest tests.py -v

# Clean up
echo "Cleaning up..."
docker stop llm-agent
docker rm llm-agent
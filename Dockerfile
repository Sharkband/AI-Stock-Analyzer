# Multi-stage Dockerfile for FastAPI + React

# Stage 1: Build React frontend
FROM node:18-alpine as react-build
WORKDIR /app

# Copy package files from frontend directory
COPY frontend/src/react/package*.json ./
RUN npm install

# Copy frontend source code
COPY frontend/src/react/ ./

# Build the React app
RUN npm run build

# Stage 2: FastAPI backend with static files
FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir "numpy<2.0.0"

# Install PyTorch CPU version first
RUN pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory structure
COPY backend/ ./

# Create necessary directories
RUN mkdir -p saved_models

# Set Python path to include the app directory
ENV PYTHONPATH=/app

# Copy built React app from previous stage to serve static files
COPY --from=react-build /app/dist ./static

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
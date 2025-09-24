# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed for FastRTC and ML models
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    libsndfile1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements files into the container at /app
COPY requirements.txt .
COPY fastrtc_requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir ensures we don't store the pip cache, keeping the image smaller
# --upgrade pip ensures we have the latest pip version
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r fastrtc_requirements.txt && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application code into the container at /app
# This includes main.py, OpenAIServer.py etc.
COPY . .

# Create directories for models and cache
RUN mkdir -p /app/models /app/cache

# Make port 8000 available to the world outside this container
EXPOSE 8000

ENV GOOGLE_API_KEY="AIzaSyA9rsKvAwQ3JP9WCr0ZW_fE3q832-B4I8s"
ENV GROQ_API_KEY="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"
ENV CLICKUP_API_TOKEN="pk_96732703_BMQ18G3SJ5N7TGYFFKAVDXXS0LP6QT1I"

# FastRTC CPU-specific environment variables
ENV TORCH_HOME=/app/models
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV CUDA_VISIBLE_DEVICES=""

# Run main.py when the container launches using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

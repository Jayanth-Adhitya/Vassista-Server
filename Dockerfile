# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Confirm versions (optional debug)
RUN node -v && npm -v && npx --version

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Set environment variables
ENV GOOGLE_API_KEY="AIzaSyA9rsKvAwQ3JP9WCr0ZW_fE3q832-B4I8s"
ENV GROQ_API_KEY="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"
ENV CLICKUP_API_TOKEN="pk_96732703_BMQ18G3SJ5N7TGYFFKAVDXXS0LP6QT1I"

# Run main.py using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

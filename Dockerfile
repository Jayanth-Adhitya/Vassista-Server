# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY main.py .

# Make port 7000 available to the world outside this container
EXPOSE 7000

# Define arguments for API keys that can be passed during build time
ARG CLICKUP_API_TOKEN
ARG GOOGLE_API_KEY
ARG GROQ_API_KEY

# Set environment variables in the container
# These will use the build arguments if provided, otherwise they will be unset
# It's recommended to provide these at runtime for better security and flexibility
ENV CLICKUP_API_TOKEN=${CLICKUP_API_TOKEN}
ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}
ENV GROQ_API_KEY=${GROQ_API_KEY}

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000"]

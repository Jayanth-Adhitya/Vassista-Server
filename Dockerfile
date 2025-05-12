# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir ensures we don't store the pip cache, keeping the image smaller
# --upgrade pip ensures we have the latest pip version
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# This includes main.py, OpenAIServer.py etc.
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

ENV GOOGLE_API_KEY="AIzaSyA9rsKvAwQ3JP9WCr0ZW_fE3q832-B4I8s"
ENV GROQ_API_KEY="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"
ENV CLICKUP_API_TOKEN="pk_96732703_BMQ18G3SJ5N7TGYFFKAVDXXS0LP6QT1I"

# Run main.py when the container launches using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Use official Python 3.10 slim image as the base
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Default command to run the app
CMD ["python", "main.py"]

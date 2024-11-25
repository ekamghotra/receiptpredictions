# Use the official Python image as base
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy requirements if any (we'll create this next)
COPY requirements.txt requirements.txt

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]

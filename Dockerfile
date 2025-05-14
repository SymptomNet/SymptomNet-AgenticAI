# Use the official Python image as a base
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl build-essential

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set work directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 5000

# Define the command to run your app
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:5000"]
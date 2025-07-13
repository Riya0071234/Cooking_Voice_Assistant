# --- Stage 1: Build Stage ---
# Use a full Python image to build our dependencies. This stage includes build tools.
FROM python:3.10-slim-bookworm as builder

# Set the working directory
WORKDIR /app

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies that might be needed by Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file to leverage Docker's cache
COPY requirements.txt .

# Install Python dependencies
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# --- Stage 2: Final Stage ---
# Use a lean Python image for the final application image.
FROM python:3.10-slim-bookworm

# Set the working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Add the project's source directory to the Python path
ENV PYTHONPATH=/app

# Copy the pre-compiled wheel files from the builder stage
COPY --from=builder /app/wheels /wheels

# Install the dependencies from the wheels
RUN pip install --no-cache /wheels/*

# Copy the application source code and configuration
COPY src/ /app/src
COPY scripts/ /app/scripts
COPY config/ /app/config

# Define the entrypoint for the container.
# This makes the container execute our pipeline runner by default.
ENTRYPOINT ["python", "scripts/pipeline_runner.py"]

# Default command can be used to pass arguments, e.g., --skip-vision
CMD [""]
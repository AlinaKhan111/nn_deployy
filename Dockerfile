# Use Python 3.10 as the base image
FROM python:3.10.14-bookworm

# Upgrade pip
RUN pip install --upgrade pip

# Copy the entire source directory to /app/src
COPY src /app/src

# Set the working directory to /app
WORKDIR /app

# Set secure permissions
RUN chmod -R 755 /app/src

# Install the required Python packages
RUN pip install -r /app/src/requirements.txt

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=${PYTHONPATH}:/app/src

# Run the training script
RUN python3 /app/src/train_pipeline.py

# Keep the container running and ready for predictions
CMD tail -f /dev/null

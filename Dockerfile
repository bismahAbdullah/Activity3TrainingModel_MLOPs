# Use official Python runtime as base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python script into the container at /app
COPY trainModel.py /app/trainModel.py

# Install any needed dependencies specified in requirements.txt
# If you have requirements.txt, uncomment the following line:
COPY requirement.txt /app/
RUN pip install --no-cache-dir -r requirement.txt

# Expose port 5000 to the outside world
EXPOSE 5000

# Run the Flask app when the container launches
CMD ["python", "trainModel.py"]

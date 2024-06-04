# Use the official ROCm TensorFlow Docker image as a base image
FROM rocm/tensorflow:latest

# Set the working directory in the container
WORKDIR /workspace

# Copy the project files into the container
COPY . /workspace

# Install any necessary dependencies
# RUN apt-get update && apt-get install -y nano vim

# Optionally, you can install Python packages listed in a requirements.txt file
RUN pip install -r requirements.txt

# Set environment variables if needed
# ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
# ENV ROCM_PATH=/opt/rocm
# ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH


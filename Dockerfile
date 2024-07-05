FROM python:3.10-slim

RUN apt-get update && apt install curl zip git build-essentials -y

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .
RUN sh setup_layoutlm_re.sh

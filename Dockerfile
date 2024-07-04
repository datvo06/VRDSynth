FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

python download_hf_models.py

sh rq1.sh
sh rq1_2_prep.sh
sh rq1_extended.sh
sh rq2.sh

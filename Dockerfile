FROM python:3.9-slim-buster

RUN apt-get update \
  && apt-get -y install tesseract-ocr ffmpeg libsm6 libxext6

WORKDIR /opt/app

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY file_reader file_reader
COPY information_extraction information_extraction
COPY layout_extraction layout_extraction
COPY utils utils
COPY models models
COPY app.py .env ./

CMD ['python', 'app.py']

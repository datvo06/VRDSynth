FROM python:3.9-slim-buster

RUN apt-get update \
  && apt-get -y install tesseract-ocr ffmpeg libsm6 libxext6

WORKDIR /opt/app

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY file_reader file_reader
COPY layout_extraction layout_extraction
COPY post_process post_process
COPY utils utils 
COPY app.py ./

EXPOSE 9000

CMD ['python', 'app.py']

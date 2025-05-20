FROM python:3.11-slim

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

ENV TESSERACT_CMD=/usr/bin/tesseract

ENV PORT 8080

CMD exec gunicorn --bind :$PORT --workers 2 --threads 2 --timeout 0 app:app
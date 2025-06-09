FROM python:3.11-slim

ENV OMP_THREAD_LIMIT=1

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

CMD exec gunicorn --bind :$PORT --workers 4 --threads 4 --timeout 0 app:app
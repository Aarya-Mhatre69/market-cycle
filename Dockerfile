FROM python:3.12-alpine

WORKDIR /app

RUN apk add --no-cache \
    curl \
    gcc \
    musl-dev \
    python3-dev \
    libffi-dev

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
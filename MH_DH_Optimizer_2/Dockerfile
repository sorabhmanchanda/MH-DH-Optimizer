# MH-DH Optimizer — Railway / container deploy
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY optimizer ./optimizer

# Railway sets PORT; default 8000 for local `docker run -p 8000:8000`
CMD sh -c 'exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"'

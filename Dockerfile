FROM python:3.11-slim

# ffmpeg is required at runtime for audio extraction from uploaded videos
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps first so this layer caches across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static

ENV PYTHONUNBUFFERED=1 \
    TMP_DIR=/tmp/consent

EXPOSE 8000

# Render injects $PORT; default to 8000 for local `docker run`.
# Long timeout because Gemini video analysis can run for minutes.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 600"]

# SwarGAN Studio API service image.
#
# Builds a CPU image by default. For GPU, base on an nvidia/cuda image and
# install the matching CUDA torch wheels instead.
FROM python:3.11-slim

# System deps: ffmpeg (Demucs / yt-dlp), build tools (pyworld), git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
        "torch>=2.8.0" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        fastapi uvicorn python-multipart \
        numpy scipy librosa soundfile pyworld matplotlib

# Copy the application code.
COPY . .

ENV SWARGAN_STORAGE_DIR=/data \
    SWARGAN_LOG_LEVEL=INFO
RUN mkdir -p /data

EXPOSE 8000

# create_app is a factory, hence --factory.
CMD ["uvicorn", "service.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]

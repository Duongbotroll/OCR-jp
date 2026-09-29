# =============================================================================
# Japanese Handwritten OCR - Docker image
#
# CPU (mặc định):
#   docker build -t jp-handwritten-ocr .
#   docker run --rm -p 8501:8501 --env-file .env jp-handwritten-ocr
#
# GPU (NVIDIA, cần NVIDIA Container Toolkit):
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
#                -t jp-handwritten-ocr:gpu .
#   docker run --rm --gpus all -p 8501:8501 --env-file .env \
#              -v hf_cache:/app/.cache/huggingface jp-handwritten-ocr:gpu
# =============================================================================
FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Thư viện hệ thống tối thiểu cho OpenCV headless + curl cho healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài torch trước từ index phù hợp (CPU/CUDA) để tận dụng Docker layer cache
RUN pip install torch torchvision --index-url "${TORCH_INDEX_URL}"

COPY requirements.txt requirements-gpu.txt ./
RUN pip install -r requirements-gpu.txt

COPY . .

# Chạy bằng user không phải root (best practice bảo mật)
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/.cache/huggingface /app/outputs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

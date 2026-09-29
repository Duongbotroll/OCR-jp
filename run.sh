#!/usr/bin/env bash
# =============================================================================
# run.sh - Tự động cài đặt môi trường và khởi chạy ứng dụng
#
# Cách dùng:
#   ./run.sh            # Cài đặt (nếu cần) + chạy Streamlit
#   ./run.sh install    # Chỉ cài đặt môi trường
#   ./run.sh start      # Chỉ chạy (bỏ qua cài đặt)
#   ./run.sh docker     # Build & chạy bằng Docker
#
# Biến môi trường tùy chọn:
#   PYTHON_BIN=python3.11  VENV_DIR=.venv  PORT=8501  HOST=0.0.0.0
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121   (cài torch bản CUDA)
#   INSTALL_MODE=full   (mặc định: có PyTorch để chạy VLM trên máy)
#   INSTALL_MODE=lite   (không PyTorch - chỉ dùng OCR qua API, vd Gemini)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
PORT="${PORT:-8501}"
HOST="${HOST:-localhost}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
IMAGE_NAME="${IMAGE_NAME:-jp-handwritten-ocr}"
INSTALL_MODE="${INSTALL_MODE:-full}"

log()  { printf '\033[1;34m[run.sh]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[run.sh]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[run.sh] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

check_python() {
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Không tìm thấy ${PYTHON_BIN}. Hãy cài Python >= 3.10."
    "${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
        || die "Cần Python >= 3.10 (hiện tại: $("${PYTHON_BIN}" --version 2>&1))."
    log "Python: $("${PYTHON_BIN}" --version 2>&1)"
}

setup_env_file() {
    if [[ ! -f .env ]]; then
        cp .env.example .env
        warn "Đã tạo .env từ .env.example - hãy điền API key trước khi dùng LLM post-correction."
    fi
}

install() {
    check_python
    if [[ ! -d "${VENV_DIR}" ]]; then
        log "Tạo virtual environment tại ${VENV_DIR}"
        "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    log "Nâng cấp pip"
    python -m pip install --upgrade pip wheel
    if [[ "${INSTALL_MODE}" == "lite" ]]; then
        log "INSTALL_MODE=lite: chỉ cài requirements.txt (OCR qua API, không PyTorch)"
        pip install -r requirements.txt
    else
        if [[ -n "${TORCH_INDEX_URL}" ]]; then
            log "Cài torch từ ${TORCH_INDEX_URL}"
            pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
        fi
        log "INSTALL_MODE=full: cài requirements-gpu.txt (có PyTorch + Transformers)"
        pip install -r requirements-gpu.txt
    fi
    setup_env_file
    mkdir -p outputs
    log "Cài đặt hoàn tất ✅"
}

start() {
    [[ -d "${VENV_DIR}" ]] || die "Chưa có ${VENV_DIR}. Chạy './run.sh install' trước."
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    setup_env_file
    log "Khởi chạy Streamlit tại http://${HOST}:${PORT}"
    exec streamlit run app.py --server.port "${PORT}" --server.address "${HOST}"
}

docker_run() {
    command -v docker >/dev/null 2>&1 || die "Không tìm thấy Docker."
    setup_env_file
    local build_args=()
    local run_args=()
    if [[ -n "${TORCH_INDEX_URL}" ]]; then
        build_args+=(--build-arg "TORCH_INDEX_URL=${TORCH_INDEX_URL}")
        run_args+=(--gpus all)
    fi
    log "Build image ${IMAGE_NAME}"
    docker build ${build_args[@]+"${build_args[@]}"} -t "${IMAGE_NAME}" .
    log "Chạy container tại http://localhost:${PORT}"
    exec docker run --rm ${run_args[@]+"${run_args[@]}"} -p "${PORT}:8501" --env-file .env \
        -v jp_ocr_hf_cache:/app/.cache/huggingface "${IMAGE_NAME}"
}

case "${1:-all}" in
    install) install ;;
    start)   start ;;
    docker)  docker_run ;;
    all)     install && start ;;
    -h|--help) sed -n '2,18p' "$0" ;;
    *) die "Lệnh không hợp lệ: $1 (dùng: install | start | docker | --help)" ;;
esac

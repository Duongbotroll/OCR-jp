"""Quản lý cấu hình tập trung cho toàn bộ dự án Japanese Handwritten OCR.

Mọi tham số đều được đọc từ biến môi trường (file ``.env``) để tuân thủ nguyên tắc
12-Factor App: cùng một mã nguồn có thể chạy ở môi trường dev, staging và production
mà chỉ cần thay đổi cấu hình, không cần sửa code.

Cách dùng:
    >>> from config import load_settings
    >>> settings = load_settings()
    >>> settings.ocr.backend
    'qwen2.5-vl'
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Nạp biến môi trường từ file .env (nếu có). Biến đã tồn tại trong shell sẽ được ưu tiên.
load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Danh sách backend OCR được hỗ trợ. Key này dùng thống nhất trong config, UI và factory.
SUPPORTED_OCR_BACKENDS: tuple[str, ...] = (
    "qwen2.5-vl",
    "glm-ocr",
    "got-ocr2",
    "openai-compatible",
)

# Model mặc định trên Hugging Face Hub cho từng backend chạy local.
# Lưu ý: luôn kiểm tra model card để biết phiên bản transformers tối thiểu.
DEFAULT_MODEL_IDS: dict[str, str] = {
    "qwen2.5-vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "glm-ocr": "zai-org/GLM-OCR",
    "got-ocr2": "stepfun-ai/GOT-OCR-2.0-hf",
    "openai-compatible": "",
}

SUPPORTED_LLM_PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "none")

# Model LLM mặc định cho bước hậu xử lý (có thể ghi đè qua LLM_MODEL).
DEFAULT_LLM_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "none": "",
}

SUPPORTED_TEXT_DIRECTIONS: tuple[str, ...] = ("auto", "horizontal", "vertical")
SUPPORTED_OCR_MODES: tuple[str, ...] = ("page", "block")


# ---------------------------------------------------------------------------
# Hàm tiện ích đọc biến môi trường có kiểu dữ liệu rõ ràng
# ---------------------------------------------------------------------------
def _env_str(key: str, default: str = "") -> str:
    """Đọc biến môi trường dạng chuỗi, loại bỏ khoảng trắng thừa."""
    value = os.getenv(key)
    return default if value is None else value.strip()


def _env_int(key: str, default: int) -> int:
    """Đọc biến môi trường dạng số nguyên, báo lỗi rõ ràng nếu sai định dạng."""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Biến môi trường {key}={raw!r} phải là số nguyên.") from exc


def _env_float(key: str, default: float) -> float:
    """Đọc biến môi trường dạng số thực."""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Biến môi trường {key}={raw!r} phải là số thực.") from exc


def _env_bool(key: str, default: bool) -> bool:
    """Đọc biến môi trường dạng boolean (true/false, 1/0, yes/no, on/off)."""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Các nhóm cấu hình
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessConfig:
    """Cấu hình tầng 1: tiền xử lý ảnh và phân tích bố cục (layout)."""

    max_side: int = 2048  # Cạnh dài tối đa của ảnh sau khi resize (pixel)
    denoise: bool = True  # Khử nhiễu Non-Local Means
    denoise_strength: int = 10  # Tham số h của fastNlMeansDenoising
    clahe: bool = True  # Tăng tương phản cục bộ bằng CLAHE
    deskew: bool = True  # Tự động chỉnh nghiêng
    max_skew_angle: float = 10.0  # Góc nghiêng tối đa cần tìm (độ)
    crop_borders: bool = True  # Cắt bỏ viền trống quanh vùng chữ
    binarize_for_ocr: bool = False  # Đưa ảnh nhị phân (thay vì ảnh xám) vào VLM
    layout_analysis: bool = True  # Phát hiện các khối văn bản
    min_block_area: int = 400  # Diện tích tối thiểu của một khối văn bản (pixel^2)
    text_direction: str = "auto"  # auto | horizontal (横書き) | vertical (縦書き)

    def __post_init__(self) -> None:
        if self.text_direction not in SUPPORTED_TEXT_DIRECTIONS:
            raise ValueError(
                f"text_direction phải thuộc {SUPPORTED_TEXT_DIRECTIONS}, "
                f"nhận được {self.text_direction!r}."
            )
        if self.max_side < 256:
            raise ValueError("max_side phải >= 256 để giữ đủ chi tiết nét chữ Kanji.")


@dataclass(frozen=True)
class OCRConfig:
    """Cấu hình tầng 2: Vision-Language OCR Engine."""

    backend: str = "qwen2.5-vl"
    model_id: str = DEFAULT_MODEL_IDS["qwen2.5-vl"]
    device: str = "auto"  # auto | cuda | cuda:0 | mps | cpu
    dtype: str = "auto"  # auto | float32 | float16 | bfloat16
    mode: str = "page"  # page: OCR cả trang | block: OCR từng khối layout
    max_new_tokens: int = 2048
    repetition_penalty: float = 1.05  # Giảm hiện tượng VLM lặp vô hạn một cụm ký tự
    trust_remote_code: bool = False
    cache_dir: str = ""  # Thư mục cache weights của Hugging Face (rỗng = mặc định)
    qwen_max_pixels: int = 1280 * 28 * 28  # Giới hạn số pixel cho Qwen2.5-VL
    api_base_url: str = "http://localhost:8000/v1"  # Endpoint OpenAI-compatible (vLLM...)
    api_key: str = ""
    api_model: str = ""
    api_timeout: float = 120.0

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_OCR_BACKENDS:
            raise ValueError(
                f"OCR backend phải thuộc {SUPPORTED_OCR_BACKENDS}, nhận được {self.backend!r}."
            )
        if self.mode not in SUPPORTED_OCR_MODES:
            raise ValueError(f"OCR mode phải thuộc {SUPPORTED_OCR_MODES}.")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens phải > 0.")


@dataclass(frozen=True)
class PostCorrectionConfig:
    """Cấu hình tầng 3: LLM hậu xử lý sửa lỗi Kanji/Kana theo ngữ cảnh."""

    provider: str = "openai"  # openai | anthropic | none
    model: str = DEFAULT_LLM_MODELS["openai"]
    api_key: str = ""
    base_url: str = ""  # Dùng cho server OpenAI-compatible (vLLM, Ollama, Azure...)
    temperature: float = 0.0  # 0 để kết quả ổn định, hạn chế "sáng tạo"
    max_tokens: int = 4096
    timeout: float = 60.0
    max_edit_ratio: float = 0.25  # Ngưỡng chống hallucination: sửa quá nhiều => từ chối
    max_chars_per_chunk: int = 1500  # Chia văn bản dài thành nhiều đoạn gửi LLM

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"LLM provider phải thuộc {SUPPORTED_LLM_PROVIDERS}, "
                f"nhận được {self.provider!r}."
            )
        if not 0.0 < self.max_edit_ratio <= 1.0:
            raise ValueError("max_edit_ratio phải nằm trong khoảng (0, 1].")
        if self.max_chars_per_chunk < 100:
            raise ValueError("max_chars_per_chunk phải >= 100.")

    @property
    def enabled(self) -> bool:
        """Bước hậu xử lý chỉ bật khi đã chọn provider và có model."""
        return self.provider != "none" and bool(self.model)


@dataclass(frozen=True)
class Settings:
    """Cấu hình tổng của ứng dụng, gom tất cả các tầng xử lý."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    post: PostCorrectionConfig = field(default_factory=PostCorrectionConfig)
    log_level: str = "INFO"
    output_dir: Path = PROJECT_ROOT / "outputs"
    pdf_dpi: int = 200  # Độ phân giải khi render PDF thành ảnh
    max_pdf_pages: int = 20  # Giới hạn số trang để tránh quá tải GPU/RAM


def _resolve_llm_api_key(provider: str) -> str:
    """Lấy API key theo provider; ưu tiên LLM_API_KEY nếu được khai báo chung."""
    generic = _env_str("LLM_API_KEY")
    if generic:
        return generic
    if provider == "openai":
        return _env_str("OPENAI_API_KEY")
    if provider == "anthropic":
        return _env_str("ANTHROPIC_API_KEY")
    return ""


def load_settings() -> Settings:
    """Tạo đối tượng ``Settings`` từ biến môi trường.

    Returns:
        Settings: cấu hình đã được validate, sẵn sàng truyền vào pipeline.

    Raises:
        ValueError: khi một biến môi trường có giá trị không hợp lệ.
    """
    preprocess = PreprocessConfig(
        max_side=_env_int("PREPROCESS_MAX_SIDE", 2048),
        denoise=_env_bool("PREPROCESS_DENOISE", True),
        denoise_strength=_env_int("PREPROCESS_DENOISE_STRENGTH", 10),
        clahe=_env_bool("PREPROCESS_CLAHE", True),
        deskew=_env_bool("PREPROCESS_DESKEW", True),
        max_skew_angle=_env_float("PREPROCESS_MAX_SKEW_ANGLE", 10.0),
        crop_borders=_env_bool("PREPROCESS_CROP_BORDERS", True),
        binarize_for_ocr=_env_bool("PREPROCESS_BINARIZE_FOR_OCR", False),
        layout_analysis=_env_bool("PREPROCESS_LAYOUT_ANALYSIS", True),
        min_block_area=_env_int("PREPROCESS_MIN_BLOCK_AREA", 400),
        text_direction=_env_str("PREPROCESS_TEXT_DIRECTION", "auto").lower(),
    )

    backend = _env_str("OCR_BACKEND", "qwen2.5-vl").lower()
    ocr = OCRConfig(
        backend=backend,
        model_id=_env_str("OCR_MODEL_ID") or DEFAULT_MODEL_IDS.get(backend, ""),
        device=_env_str("OCR_DEVICE", "auto"),
        dtype=_env_str("OCR_DTYPE", "auto").lower(),
        mode=_env_str("OCR_MODE", "page").lower(),
        max_new_tokens=_env_int("OCR_MAX_NEW_TOKENS", 2048),
        repetition_penalty=_env_float("OCR_REPETITION_PENALTY", 1.05),
        trust_remote_code=_env_bool("OCR_TRUST_REMOTE_CODE", False),
        cache_dir=_env_str("HF_CACHE_DIR"),
        qwen_max_pixels=_env_int("OCR_QWEN_MAX_PIXELS", 1280 * 28 * 28),
        api_base_url=_env_str("OCR_API_BASE_URL", "http://localhost:8000/v1"),
        api_key=_env_str("OCR_API_KEY"),
        api_model=_env_str("OCR_API_MODEL"),
        api_timeout=_env_float("OCR_API_TIMEOUT", 120.0),
    )

    provider = _env_str("LLM_PROVIDER", "openai").lower()
    post = PostCorrectionConfig(
        provider=provider,
        model=_env_str("LLM_MODEL") or DEFAULT_LLM_MODELS.get(provider, ""),
        api_key=_resolve_llm_api_key(provider),
        base_url=_env_str("OPENAI_BASE_URL") if provider == "openai" else "",
        temperature=_env_float("LLM_TEMPERATURE", 0.0),
        max_tokens=_env_int("LLM_MAX_TOKENS", 4096),
        timeout=_env_float("LLM_TIMEOUT", 60.0),
        max_edit_ratio=_env_float("LLM_MAX_EDIT_RATIO", 0.25),
        max_chars_per_chunk=_env_int("LLM_MAX_CHARS_PER_CHUNK", 1500),
    )

    output_dir = Path(_env_str("OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))

    return Settings(
        preprocess=preprocess,
        ocr=ocr,
        post=post,
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        output_dir=output_dir,
        pdf_dpi=_env_int("PDF_DPI", 200),
        max_pdf_pages=_env_int("MAX_PDF_PAGES", 20),
    )


if __name__ == "__main__":
    # Chạy `python config.py` để in ra cấu hình hiện tại (API key được che bớt).
    current = load_settings()
    masked_post = current.post.api_key[:4] + "***" if current.post.api_key else "(empty)"
    print(f"Preprocess : {current.preprocess}")
    print(f"OCR        : backend={current.ocr.backend}, model={current.ocr.model_id}")
    print(f"LLM        : provider={current.post.provider}, model={current.post.model}, "
          f"api_key={masked_post}")

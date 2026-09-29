"""Tiện ích dùng chung: logging, đọc/ghi ảnh & PDF, đo thời gian và chỉ số đánh giá CER.

Module này KHÔNG phụ thuộc vào torch/transformers để có thể dùng trong mọi môi trường
(CI, unit test, container CPU-only...).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from types import TracebackType
from typing import Any, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageSequence

# Kiểu dữ liệu đầu vào ảnh được chấp nhận bởi hàm load_image
ImageSource = Union[str, Path, bytes, np.ndarray, Image.Image]

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)
SUPPORTED_DOCUMENT_EXTENSIONS: frozenset[str] = SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CODE_FENCE_PATTERN = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(name: str = "jp_ocr", level: str = "INFO") -> logging.Logger:
    """Tạo logger với format thống nhất, tránh gắn handler trùng lặp khi Streamlit rerun.

    Args:
        name: Tên logger (thường truyền ``__name__``).
        level: Mức log: DEBUG, INFO, WARNING, ERROR.

    Returns:
        logging.Logger đã được cấu hình.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level.upper())
    return logger


# ---------------------------------------------------------------------------
# Chuyển đổi định dạng ảnh
# ---------------------------------------------------------------------------
def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Chuẩn hóa mọi ảnh numpy (xám, BGR, BGRA) về dạng BGR 3 kênh uint8."""
    if image.dtype != np.uint8:
        # Chuẩn hóa ảnh 16-bit / float về uint8 để OpenCV xử lý thống nhất
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Chuyển ảnh PIL (RGB) sang numpy BGR để dùng với OpenCV."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Chuyển ảnh numpy (BGR hoặc xám) sang PIL RGB để đưa vào Vision-Language Model."""
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def encode_png_bytes(image: np.ndarray) -> bytes:
    """Mã hóa ảnh numpy thành bytes PNG (không mất dữ liệu) để gửi qua API."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Không thể mã hóa ảnh sang PNG.")
    return buffer.tobytes()


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """Thu nhỏ ảnh sao cho cạnh dài nhất không vượt quá ``max_side`` (giữ tỉ lệ).

    Chỉ thu nhỏ, không phóng to: phóng to không thêm thông tin mà chỉ tăng chi phí tính toán.
    """
    if max_side <= 0:
        return image
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Đọc ảnh và tài liệu
# ---------------------------------------------------------------------------
def load_image(source: ImageSource) -> np.ndarray:
    """Đọc ảnh từ nhiều nguồn khác nhau và trả về numpy BGR.

    Ảnh chụp từ điện thoại thường có thẻ EXIF Orientation; ``exif_transpose`` xoay ảnh
    về đúng chiều trước khi xử lý, tránh việc OCR nhận ảnh bị xoay 90 độ.

    Args:
        source: Đường dẫn file, bytes, numpy array hoặc PIL Image.

    Returns:
        numpy.ndarray dạng BGR uint8.

    Raises:
        FileNotFoundError: nếu đường dẫn không tồn tại.
        ValueError: nếu dữ liệu không phải ảnh hợp lệ.
    """
    if isinstance(source, np.ndarray):
        return ensure_bgr(source.copy())
    try:
        if isinstance(source, Image.Image):
            pil_image = source
        elif isinstance(source, (bytes, bytearray)):
            pil_image = Image.open(io.BytesIO(source))
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy file ảnh: {path}")
            pil_image = Image.open(path)
        pil_image = ImageOps.exif_transpose(pil_image)
        return pil_to_bgr(pil_image)
    except FileNotFoundError:
        raise
    except Exception as exc:  # PIL có thể ném nhiều loại lỗi khác nhau
        raise ValueError(f"Dữ liệu không phải ảnh hợp lệ: {exc}") from exc


def _load_pdf_pages(data: bytes, pdf_dpi: int, max_pages: int) -> list[np.ndarray]:
    """Render từng trang PDF thành ảnh BGR bằng PyMuPDF (không cần Poppler)."""
    import fitz  # PyMuPDF - import trễ để không bắt buộc khi chỉ xử lý ảnh

    pages: list[np.ndarray] = []
    with fitz.open(stream=data, filetype="pdf") as document:
        for index, page in enumerate(document):
            if index >= max_pages:
                break
            pixmap = page.get_pixmap(dpi=pdf_dpi)
            array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            if pixmap.n == 1:
                pages.append(cv2.cvtColor(array, cv2.COLOR_GRAY2BGR))
            elif pixmap.n == 4:
                pages.append(cv2.cvtColor(array, cv2.COLOR_RGBA2BGR))
            else:
                pages.append(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    return pages


def load_document_pages(
    data: bytes, filename: str, pdf_dpi: int = 200, max_pages: int = 20
) -> list[np.ndarray]:
    """Đọc một tài liệu (ảnh đơn, TIFF nhiều trang, hoặc PDF) thành danh sách trang BGR.

    Args:
        data: Nội dung file dạng bytes.
        filename: Tên file gốc, dùng để nhận diện định dạng qua phần mở rộng.
        pdf_dpi: Độ phân giải render PDF (200 DPI đủ rõ cho nét chữ viết tay).
        max_pages: Số trang tối đa được xử lý.

    Returns:
        Danh sách ảnh BGR, mỗi phần tử là một trang.

    Raises:
        ValueError: nếu định dạng không được hỗ trợ hoặc file rỗng.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(
            f"Định dạng {suffix!r} không được hỗ trợ. "
            f"Hỗ trợ: {', '.join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))}"
        )
    if not data:
        raise ValueError("File rỗng.")

    if suffix == ".pdf":
        pages = _load_pdf_pages(data, pdf_dpi=pdf_dpi, max_pages=max_pages)
    elif suffix in {".tif", ".tiff"}:
        # TIFF từ máy scan văn phòng Nhật thường chứa nhiều trang
        pages = []
        with Image.open(io.BytesIO(data)) as tiff:
            for index, frame in enumerate(ImageSequence.Iterator(tiff)):
                if index >= max_pages:
                    break
                pages.append(pil_to_bgr(frame.copy()))
    else:
        pages = [load_image(data)]

    if not pages:
        raise ValueError("Không đọc được trang nào từ tài liệu.")
    return pages


# ---------------------------------------------------------------------------
# I/O kết quả
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    """Tính SHA-256 của dữ liệu, dùng làm khóa cache hoặc định danh tài liệu."""
    return hashlib.sha256(data).hexdigest()


def save_json(data: Any, path: Union[str, Path]) -> Path:
    """Lưu dữ liệu ra file JSON UTF-8 (giữ nguyên ký tự tiếng Nhật, không escape)."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str)
    return output_path


def strip_code_fences(text: str) -> str:
    """Loại bỏ khối Markdown ```...``` mà một số VLM/LLM tự thêm vào output."""
    match = _CODE_FENCE_PATTERN.match(text)
    return match.group(1) if match else text


# ---------------------------------------------------------------------------
# Đo thời gian
# ---------------------------------------------------------------------------
class Timer:
    """Context manager đo thời gian thực thi (giây) với độ chính xác cao.

    Ví dụ:
        >>> with Timer() as timer:
        ...     do_something()
        >>> print(f"{timer.elapsed:.2f}s")
    """

    def __init__(self) -> None:
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.elapsed = time.perf_counter() - self.start


# ---------------------------------------------------------------------------
# Chỉ số đánh giá
# ---------------------------------------------------------------------------
def normalize_for_eval(text: str) -> str:
    """Chuẩn hóa văn bản trước khi tính CER.

    - NFKC: đồng nhất ký tự full-width/half-width (ＡＢＣ -> ABC, ｶﾀｶﾅ -> カタカナ).
    - Bỏ toàn bộ khoảng trắng và xuống dòng, vì tiếng Nhật không dùng dấu cách giữa từ
      và cách ngắt dòng không phản ánh chất lượng nhận dạng ký tự.
    """
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(normalized.split())


def levenshtein_distance(source: str, target: str) -> int:
    """Khoảng cách Levenshtein ở mức ký tự (quy hoạch động, bộ nhớ O(min(n, m)))."""
    if len(source) < len(target):
        source, target = target, source
    if not target:
        return len(source)
    previous = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (source_char != target_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """Tính Character Error Rate (CER) - chỉ số chuẩn cho OCR tiếng Nhật.

    CER = (S + D + I) / N, với N là số ký tự của văn bản tham chiếu (ground truth).

    Args:
        reference: Văn bản đúng (ground truth).
        hypothesis: Văn bản do hệ thống nhận dạng.
        normalize: Có chuẩn hóa NFKC và bỏ khoảng trắng trước khi so sánh hay không.

    Returns:
        CER dạng số thực (0.0 là hoàn hảo; có thể > 1.0 nếu hypothesis dài hơn nhiều).
    """
    if normalize:
        reference = normalize_for_eval(reference)
        hypothesis = normalize_for_eval(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein_distance(reference, hypothesis) / len(reference)

"""Tầng 2 - Vision-Language OCR Engine cho chữ viết tay Kanji/Kana.

Thiết kế theo Strategy Pattern + Factory: mỗi model là một ``Backend`` độc lập, cùng
interface ``recognize(image, prompt) -> str``. Nhờ vậy có thể thay model SOTA mới
chỉ bằng cách viết thêm một class, không cần sửa pipeline hay UI.

Backend hỗ trợ:
    - ``qwen2.5-vl``        : Qwen2.5-VL (3B/7B/72B) - VLM đa năng, tiếng Nhật tốt.
    - ``glm-ocr``           : GLM-OCR (Z.ai) - model OCR chuyên dụng, nhẹ.
    - ``got-ocr2``          : GOT-OCR 2.0 - OCR end-to-end, rất nhanh.
    - ``openai-compatible`` : Gọi VLM qua API chuẩn OpenAI (vLLM, SGLang, LMDeploy...).

Các thư viện nặng (torch, transformers, openai) được import trễ bên trong từng backend.
"""

from __future__ import annotations

import base64
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from config import OCRConfig
from src.preprocessor import PreprocessResult
from src.utils import Timer, bgr_to_pil, encode_png_bytes, setup_logger, strip_code_fences

logger = setup_logger(__name__)

# Prompt mặc định viết bằng tiếng Nhật: VLM tuân thủ chỉ dẫn tốt hơn khi ngôn ngữ prompt
# trùng với ngôn ngữ đầu ra mong muốn. Ký tự 〓 (ゲタ記号) là quy ước của ngành in ấn
# Nhật Bản để đánh dấu ký tự không đọc được.
DEFAULT_OCR_PROMPT: str = (
    "あなたは日本語の手書き文書を専門とする高精度OCRエンジンです。"
    "画像内の文字（漢字・ひらがな・カタカナ・英数字・記号）を、見えるとおりに正確に"
    "書き起こしてください。\n"
    "ルール:\n"
    "1. 書き起こしたテキストのみを出力し、要約・翻訳・説明・前置きは一切書かない。\n"
    "2. 改行は原文の行の区切りに合わせる。縦書きの場合は右の列から左の列へ、"
    "各列を上から下へ読む。\n"
    "3. 誤字や不自然な表現も修正せず、書かれているとおりに出力する。\n"
    "4. 判読できない文字は推測せず「〓」で表す。\n"
    "5. Markdownやコードブロックで囲まない。"
)

# GLM-OCR được huấn luyện với các prompt tác vụ ngắn (xem model card để cập nhật).
GLM_OCR_TASK_PROMPT: str = "Text Recognition:"

_MULTI_BLANK_LINES = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Kết quả OCR
# ---------------------------------------------------------------------------
@dataclass
class OCRResult:
    """Kết quả nhận dạng của một trang tài liệu."""

    text: str
    backend: str
    model_id: str
    mode: str
    elapsed_seconds: float
    block_texts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Chuyển sang dict để xuất JSON."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Hàm hỗ trợ torch (import trễ)
# ---------------------------------------------------------------------------
def _resolve_device(requested: str) -> str:
    """Chọn thiết bị tính toán: CUDA > Apple MPS > CPU khi ``requested == 'auto'``."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype_name: str, device: str) -> Any:
    """Chọn kiểu dữ liệu tensor phù hợp với phần cứng.

    - GPU Ampere trở lên: bfloat16 (ổn định số học, tiết kiệm 50% VRAM so với float32).
    - GPU cũ: float16. Apple MPS: float16. CPU: float32 (CPU xử lý half-precision chậm).
    """
    import torch

    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype_name != "auto":
        if dtype_name not in mapping:
            raise ValueError(f"dtype {dtype_name!r} không hợp lệ. Chọn: auto, {list(mapping)}")
        return mapping[dtype_name]
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Các backend
# ---------------------------------------------------------------------------
class BaseOCRBackend(ABC):
    """Interface chung cho mọi OCR backend."""

    name: str = "base"

    def __init__(self, config: OCRConfig) -> None:
        self.config = config

    @property
    def model_id(self) -> str:
        """Định danh model đang dùng (hiển thị trong UI và kết quả JSON)."""
        return self.config.model_id

    @abstractmethod
    def recognize(self, image: np.ndarray, prompt: str = DEFAULT_OCR_PROMPT) -> str:
        """Nhận dạng văn bản trong ảnh BGR và trả về chuỗi thô (chưa hậu xử lý)."""


class HFVisionLanguageBackend(BaseOCRBackend):
    """Backend tổng quát cho các VLM dạng chat trên Hugging Face Transformers.

    Dùng ``AutoModelForImageTextToText`` + ``processor.apply_chat_template`` nên tương thích
    với phần lớn VLM hiện đại (Qwen2.5-VL, GLM-OCR, ...), miễn là transformers đủ mới.
    """

    name = "hf-vlm"

    def __init__(
        self, config: OCRConfig, processor_kwargs: Optional[dict[str, Any]] = None
    ) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.device = _resolve_device(config.device)
        self.dtype = _resolve_dtype(config.dtype, self.device)
        cache_dir = config.cache_dir or None
        logger.info(
            "Loading %s on %s (%s) - lần đầu có thể mất vài phút để tải weights...",
            config.model_id, self.device, self.dtype,
        )

        self.processor = AutoProcessor.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
            cache_dir=cache_dir,
            **(processor_kwargs or {}),
        )
        load_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "trust_remote_code": config.trust_remote_code,
            "cache_dir": cache_dir,
        }
        if self.device.startswith("cuda"):
            # device_map="auto" tự chia model lên nhiều GPU nếu một GPU không đủ VRAM
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForImageTextToText.from_pretrained(config.model_id, **load_kwargs)
        if not self.device.startswith("cuda"):
            self.model.to(self.device)
        self.model.eval()

    def _build_messages(self, prompt: str) -> list[dict[str, Any]]:
        """Tạo hội thoại dạng chat gồm một ảnh và một câu lệnh văn bản."""
        return [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]

    def recognize(self, image: np.ndarray, prompt: str = DEFAULT_OCR_PROMPT) -> str:
        """Chạy suy luận greedy (do_sample=False) để kết quả OCR mang tính tất định."""
        pil_image = bgr_to_pil(image)
        chat_text = self.processor.apply_chat_template(
            self._build_messages(prompt), tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[chat_text], images=[pil_image], return_tensors="pt", padding=True
        )
        # BatchFeature.to(dtype=...) chỉ ép kiểu tensor số thực (pixel_values),
        # input_ids vẫn giữ kiểu int.
        inputs = inputs.to(self.model.device, dtype=self.dtype)

        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                repetition_penalty=self.config.repetition_penalty,
            )
        # Chỉ giải mã phần token mới sinh ra, bỏ phần prompt
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0].strip() if decoded else ""


class QwenVLBackend(HFVisionLanguageBackend):
    """Qwen2.5-VL: giới hạn số pixel để kiểm soát số visual token và VRAM.

    Qwen2.5-VL chia ảnh thành patch 28x28; ``max_pixels`` = 1280*28*28 tương đương tối đa
    ~1280 visual token - cân bằng tốt giữa độ chi tiết nét Kanji và tốc độ.
    """

    name = "qwen2.5-vl"

    def __init__(self, config: OCRConfig) -> None:
        super().__init__(
            config,
            processor_kwargs={"min_pixels": 256 * 28 * 28, "max_pixels": config.qwen_max_pixels},
        )


class GLMOCRBackend(HFVisionLanguageBackend):
    """GLM-OCR: model OCR chuyên dụng dùng prompt tác vụ ngắn thay vì chỉ dẫn dài."""

    name = "glm-ocr"

    def recognize(self, image: np.ndarray, prompt: str = DEFAULT_OCR_PROMPT) -> str:
        # Nếu người dùng giữ prompt mặc định, chuyển sang prompt tác vụ chuẩn của GLM-OCR.
        effective_prompt = GLM_OCR_TASK_PROMPT if prompt == DEFAULT_OCR_PROMPT else prompt
        return super().recognize(image, effective_prompt)


class GOTOCR2Backend(BaseOCRBackend):
    """GOT-OCR 2.0 (bản tích hợp native trong transformers, model ``-hf``).

    GOT-OCR2 không nhận prompt tự do; tham số ``prompt`` bị bỏ qua. Lưu ý: dữ liệu huấn
    luyện chủ yếu là tiếng Anh/Trung, nên với chữ viết tay tiếng Nhật cần kỳ vọng hợp lý
    và nên bật LLM post-correction.
    """

    name = "got-ocr2"

    def __init__(self, config: OCRConfig) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.device = _resolve_device(config.device)
        self.dtype = _resolve_dtype(config.dtype, self.device)
        cache_dir = config.cache_dir or None
        logger.info("Loading %s on %s (%s)...", config.model_id, self.device, self.dtype)

        self.processor = AutoProcessor.from_pretrained(config.model_id, cache_dir=cache_dir)
        load_kwargs: dict[str, Any] = {"torch_dtype": self.dtype, "cache_dir": cache_dir}
        if self.device.startswith("cuda"):
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForImageTextToText.from_pretrained(config.model_id, **load_kwargs)
        if not self.device.startswith("cuda"):
            self.model.to(self.device)
        self.model.eval()

    def recognize(self, image: np.ndarray, prompt: str = DEFAULT_OCR_PROMPT) -> str:
        pil_image = bgr_to_pil(image)
        inputs = self.processor(pil_image, return_tensors="pt").to(
            self.model.device, dtype=self.dtype
        )
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=self.config.max_new_tokens,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""


class OpenAICompatibleVisionBackend(BaseOCRBackend):
    """Gọi VLM qua REST API chuẩn OpenAI Chat Completions.

    Phù hợp triển khai production: chạy model trên GPU server bằng vLLM
    (``vllm serve Qwen/Qwen2.5-VL-7B-Instruct``) và ứng dụng web chỉ là client nhẹ.
    Dữ liệu vẫn nằm trong hạ tầng nội bộ - quan trọng với luật APPI của Nhật Bản.
    """

    name = "openai-compatible"

    def __init__(self, config: OCRConfig) -> None:
        super().__init__(config)
        from openai import OpenAI

        if not config.api_model:
            raise ValueError("Backend openai-compatible cần khai báo OCR_API_MODEL.")
        self.client = OpenAI(
            base_url=config.api_base_url or None,
            api_key=config.api_key or "EMPTY",  # vLLM mặc định không kiểm tra key
            timeout=config.api_timeout,
        )

    @property
    def model_id(self) -> str:
        return self.config.api_model

    def recognize(self, image: np.ndarray, prompt: str = DEFAULT_OCR_PROMPT) -> str:
        encoded = base64.b64encode(encode_png_bytes(image)).decode("ascii")
        response = self.client.chat.completions.create(
            model=self.config.api_model,
            temperature=0.0,
            max_tokens=self.config.max_new_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        content = response.choices[0].message.content if response.choices else ""
        return (content or "").strip()


_BACKEND_REGISTRY: dict[str, type[BaseOCRBackend]] = {
    "qwen2.5-vl": QwenVLBackend,
    "glm-ocr": GLMOCRBackend,
    "got-ocr2": GOTOCR2Backend,
    "openai-compatible": OpenAICompatibleVisionBackend,
}


def create_ocr_backend(config: OCRConfig) -> BaseOCRBackend:
    """Factory tạo backend theo ``config.backend``.

    Raises:
        ValueError: nếu backend chưa được đăng ký.
    """
    backend_cls = _BACKEND_REGISTRY.get(config.backend)
    if backend_cls is None:
        raise ValueError(
            f"Backend {config.backend!r} chưa được hỗ trợ. Có sẵn: {list(_BACKEND_REGISTRY)}"
        )
    return backend_cls(config)


# ---------------------------------------------------------------------------
# Engine điều phối
# ---------------------------------------------------------------------------
class OCREngine:
    """Điều phối việc OCR một trang: cả trang (page) hoặc từng khối layout (block)."""

    def __init__(self, config: OCRConfig, backend: Optional[BaseOCRBackend] = None) -> None:
        self.config = config
        # Cho phép inject backend giả lập khi viết unit test (Dependency Injection)
        self.backend = backend or create_ocr_backend(config)

    @staticmethod
    def clean_output(text: str) -> str:
        """Làm sạch output của VLM: bỏ code fence, chuẩn hóa xuống dòng, bỏ dòng trống thừa."""
        cleaned = strip_code_fences(text).replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in cleaned.split("\n")]
        return _MULTI_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()

    def run(self, preprocessed: PreprocessResult, prompt: Optional[str] = None) -> OCRResult:
        """OCR một trang đã được tiền xử lý.

        - Chế độ ``page``: gửi cả trang cho VLM. VLM hiện đại tự hiểu bố cục => chính xác
          hơn và chỉ tốn một lần suy luận. Đây là chế độ khuyến nghị.
        - Chế độ ``block``: OCR từng khối theo thứ tự đọc đã tính ở tầng 1. Hữu ích cho
          trang rất lớn/dày đặc hoặc model có giới hạn độ phân giải thấp.
        """
        effective_prompt = prompt or DEFAULT_OCR_PROMPT
        block_texts: list[str] = []
        with Timer() as timer:
            if self.config.mode == "block" and preprocessed.blocks:
                for block in preprocessed.blocks:
                    crop = block.crop(preprocessed.image)
                    block_text = self.clean_output(self.backend.recognize(crop, effective_prompt))
                    block_texts.append(block_text)
                text = "\n".join(item for item in block_texts if item)
            else:
                text = self.clean_output(
                    self.backend.recognize(preprocessed.image, effective_prompt)
                )

        logger.info(
            "OCR done: backend=%s, mode=%s, chars=%d, %.2fs",
            self.backend.name, self.config.mode, len(text), timer.elapsed,
        )
        return OCRResult(
            text=text,
            backend=self.backend.name,
            model_id=self.backend.model_id,
            mode=self.config.mode,
            elapsed_seconds=round(timer.elapsed, 3),
            block_texts=block_texts,
        )

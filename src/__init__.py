"""Japanese Handwritten Document OCR & Parsing System.

Package ``src`` chứa 3 tầng của pipeline:

1. ``preprocessor``  - Tiền xử lý ảnh & phân tích layout (OpenCV).
2. ``ocr_engine``    - Nhận dạng chữ bằng Vision-Language Model (Qwen2.5-VL / GLM-OCR / GOT-OCR2).
3. ``postprocessor`` - LLM hậu xử lý, sửa lỗi Kanji/Kana dựa trên ngữ cảnh.

Các module nặng (torch, transformers) chỉ được import khi thực sự cần (lazy import),
nên việc ``import src`` luôn nhẹ và nhanh.
"""

__version__ = "1.0.0"
__author__ = "Japanese Handwritten OCR Contributors"

__all__ = ["__version__", "__author__"]

"""Tầng 3 - LLM Contextual Post-Correction: sửa lỗi Kanji/Kana dựa trên ngữ cảnh.

Vấn đề: OCR chữ viết tay tiếng Nhật hay nhầm các ký tự có hình dạng gần giống nhau
(未/末, 土/士, シ/ツ, ソ/ン, カ/力, ロ/口, ニ/二, 工/エ, ー/一, へ/ヘ...). Một mô hình ngôn ngữ
lớn hiểu ngữ cảnh câu có thể phát hiện và sửa các lỗi này.

Rủi ro: LLM có xu hướng "viết lại cho hay hơn" (hallucination / paraphrase) - điều tối kỵ
trong số hóa tài liệu pháp lý, y tế, tài chính. Vì vậy module áp dụng 3 lớp bảo vệ:

1. Prompt nghiêm ngặt: chỉ sửa lỗi nhận dạng, cấm diễn đạt lại, cấm đổi số liệu.
2. Output có cấu trúc JSON kèm danh sách từng chỗ sửa và lý do (có thể audit).
3. Edit-ratio guard: nếu mức thay đổi vượt ngưỡng, từ chối bản sửa và giữ nguyên bản gốc.
"""

from __future__ import annotations

import difflib
import html
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from config import PostCorrectionConfig
from src.utils import setup_logger, strip_code_fences

logger = setup_logger(__name__)

SYSTEM_PROMPT: str = """あなたは日本語の手書き文書をOCRした結果を校正する専門家です。
OCRエンジンが誤認識した文字だけを、文脈に基づいて最小限に修正してください。

# 修正してよいもの
- 字形が似ていることによる誤認識（例: 未↔末、土↔士、シ↔ツ、ソ↔ン、カ↔力、ロ↔口、ニ↔二、工↔エ、ー↔一、へ↔ヘ、日↔曰）
- 文脈上明らかに誤っている漢字・送り仮名・ひらがな／カタカナの取り違え
- OCRが挿入した明らかに不要な記号・重複文字
- 「〓」（判読不能文字）は、文脈から一意に確定できる場合のみ置き換えてよい

# 絶対に守るルール
- 文章の言い換え・要約・敬語の変更・語順の変更をしない
- 数字・金額・日付・電話番号・住所・人名・固有名詞は、明白な誤認識でない限り変更しない
- 原文の書き手自身の誤字か、OCRの誤認識か判断できない場合は変更しない
- 改行の位置と行数をそのまま維持する
- 内容を追加・削除しない

# 出力形式
次のJSONオブジェクトのみを出力すること（前置き・説明・Markdownは不要）:
{"corrected_text": "校正後の全文", "corrections": [{"original": "誤", "corrected": "正", "reason": "理由（簡潔に日本語で）"}]}
修正がない場合は corrected_text に原文をそのまま入れ、corrections は空配列にする。"""

USER_PROMPT_TEMPLATE: str = """## 文書の種類・背景
{context}

## OCR結果
<ocr_text>
{text}
</ocr_text>

上記のOCR結果を校正し、指定されたJSON形式のみで回答してください。"""


# ---------------------------------------------------------------------------
# Kiểu dữ liệu kết quả
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Correction:
    """Một chỗ sửa do LLM đề xuất (kèm lý do để con người kiểm tra lại)."""

    original: str
    corrected: str
    reason: str = ""


@dataclass
class PostCorrectionResult:
    """Kết quả của tầng hậu xử lý."""

    original_text: str
    corrected_text: str
    corrections: list[Correction] = field(default_factory=list)
    applied: bool = False  # True nếu LLM đã thực sự được gọi và có ít nhất một đoạn được áp dụng
    edit_ratio: float = 0.0  # Tỉ lệ thay đổi toàn văn bản (0 = không đổi)
    provider: str = "none"
    model: str = ""
    rejected_chunks: int = 0  # Số đoạn bị guard từ chối vì sửa quá nhiều
    errors: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Chuyển sang dict để xuất JSON."""
        return asdict(self)


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------
class BaseLLMClient(ABC):
    """Interface tối giản cho một LLM chat: nhận system + user prompt, trả về text."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Gọi LLM và trả về nội dung phản hồi dạng chuỗi."""


class OpenAIChatClient(BaseLLMClient):
    """Client cho OpenAI và mọi server tương thích (vLLM, Ollama, Azure OpenAI...)."""

    def __init__(self, config: PostCorrectionConfig) -> None:
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            api_key=config.api_key or None,
            base_url=config.base_url or None,
            timeout=config.timeout,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "") if response.choices else ""


class AnthropicChatClient(BaseLLMClient):
    """Client cho Anthropic Claude (Messages API)."""

    def __init__(self, config: PostCorrectionConfig) -> None:
        from anthropic import Anthropic

        self.config = config
        self.client = Anthropic(api_key=config.api_key or None, timeout=config.timeout)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )


def create_llm_client(config: PostCorrectionConfig) -> Optional[BaseLLMClient]:
    """Factory tạo LLM client; trả về None nếu hậu xử lý bị tắt."""
    if not config.enabled:
        return None
    if config.provider == "openai":
        return OpenAIChatClient(config)
    if config.provider == "anthropic":
        return AnthropicChatClient(config)
    raise ValueError(f"LLM provider không được hỗ trợ: {config.provider!r}")


# ---------------------------------------------------------------------------
# Hàm tiện ích xử lý văn bản
# ---------------------------------------------------------------------------
def split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Chia văn bản thành các đoạn <= ``max_chars`` ký tự, ưu tiên cắt tại ranh giới dòng.

    Chia đoạn giúp: (1) không vượt giới hạn context, (2) LLM tập trung hơn nên ít bỏ sót,
    (3) guard edit-ratio hoạt động ở mức chi tiết - một đoạn hỏng không kéo theo cả trang.
    """
    lines = text.split("\n")
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for line in lines:
        # Một dòng quá dài (hiếm gặp) được cắt cứng thành nhiều phần
        while len(line) > max_chars:
            if buffer:
                chunks.append("\n".join(buffer))
                buffer, buffer_len = [], 0
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        added_len = len(line) + (1 if buffer else 0)
        if buffer and buffer_len + added_len > max_chars:
            chunks.append("\n".join(buffer))
            buffer, buffer_len = [line], len(line)
        else:
            buffer.append(line)
            buffer_len += added_len
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Trích xuất object JSON từ phản hồi của LLM một cách chịu lỗi.

    Xử lý các trường hợp LLM bọc JSON trong ```json ...``` hoặc thêm câu dẫn trước/sau.

    Raises:
        ValueError: nếu không tìm thấy JSON hợp lệ.
    """
    candidate = strip_code_fences(raw.strip())
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("Phản hồi của LLM không chứa JSON.") from None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON không hợp lệ từ LLM: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("JSON từ LLM phải là một object.")
    return parsed


def compute_edit_ratio(original: str, corrected: str) -> float:
    """Tỉ lệ thay đổi ở mức ký tự: 0.0 = giống hệt, 1.0 = khác hoàn toàn."""
    if not original and not corrected:
        return 0.0
    matcher = difflib.SequenceMatcher(None, original, corrected, autojunk=False)
    return round(1.0 - matcher.ratio(), 4)


def extract_char_changes(original: str, corrected: str) -> list[dict[str, Any]]:
    """Liệt kê các thay đổi THỰC TẾ giữa hai văn bản (tính bằng diff, không tin LLM).

    Danh sách ``corrections`` do LLM tự báo cáo có thể thiếu hoặc sai; hàm này cung cấp
    nguồn sự thật có thể kiểm chứng cho người duyệt (human-in-the-loop).
    """
    matcher = difflib.SequenceMatcher(None, original, corrected, autojunk=False)
    changes: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changes.append(
            {
                "operation": tag,  # replace | delete | insert
                "position": i1,
                "original": original[i1:i2],
                "corrected": corrected[j1:j2],
            }
        )
    return changes


def render_diff_html(original: str, corrected: str) -> str:
    """Tạo HTML tô màu khác biệt: đỏ gạch ngang = bị xóa, xanh = được thêm."""
    matcher = difflib.SequenceMatcher(None, original, corrected, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old = html.escape(original[i1:i2]).replace("\n", "<br>")
        new = html.escape(corrected[j1:j2]).replace("\n", "<br>")
        if tag == "equal":
            parts.append(old)
            continue
        if old:
            parts.append(
                "<del style='background:#ffd7d5;color:#82071e;text-decoration:line-through;'>"
                f"{old}</del>"
            )
        if new:
            parts.append(
                "<ins style='background:#ccffd8;color:#055d20;text-decoration:none;"
                f"font-weight:600;'>{new}</ins>"
            )
    body = "".join(parts)
    return (
        "<div style=\"font-family:'Noto Sans JP','Hiragino Sans','Yu Gothic',sans-serif;"
        f"font-size:1.05rem;line-height:1.9;white-space:normal;\">{body}</div>"
    )


# ---------------------------------------------------------------------------
# Bộ hậu xử lý chính
# ---------------------------------------------------------------------------
class JapanesePostCorrector:
    """Sửa lỗi OCR tiếng Nhật bằng LLM, có cơ chế chống hallucination."""

    def __init__(
        self, config: PostCorrectionConfig, client: Optional[BaseLLMClient] = None
    ) -> None:
        self.config = config
        # Cho phép inject client giả lập trong unit test
        self.client = client if client is not None else create_llm_client(config)

    @property
    def enabled(self) -> bool:
        """Chỉ hoạt động khi có client hợp lệ."""
        return self.client is not None

    @staticmethod
    def _parse_corrections(payload: dict[str, Any]) -> list[Correction]:
        """Đọc danh sách corrections, bỏ qua phần tử sai định dạng thay vì làm hỏng cả kết quả."""
        items = payload.get("corrections") or []
        corrections: list[Correction] = []
        if not isinstance(items, list):
            return corrections
        for item in items:
            if not isinstance(item, dict):
                continue
            original = str(item.get("original", ""))
            corrected = str(item.get("corrected", ""))
            if original == corrected:
                continue
            corrections.append(
                Correction(
                    original=original,
                    corrected=corrected,
                    reason=str(item.get("reason", "")),
                )
            )
        return corrections

    def correct(self, text: str, context: str = "") -> PostCorrectionResult:
        """Hậu xử lý văn bản OCR.

        Args:
            text: Văn bản thô từ OCR engine.
            context: Mô tả loại tài liệu (vd: "請求書", "医療問診票", "申込書") giúp LLM
                chọn đúng từ vựng chuyên ngành.

        Returns:
            PostCorrectionResult. Khi có lỗi mạng/API, trả về văn bản gốc (fail-safe)
            và ghi lỗi vào ``errors`` - pipeline không bao giờ bị dừng vì LLM.
        """
        if not text.strip():
            return PostCorrectionResult(
                original_text=text, corrected_text=text, message="Văn bản rỗng, bỏ qua."
            )
        if not self.enabled:
            return PostCorrectionResult(
                original_text=text, corrected_text=text,
                message="LLM post-correction đang tắt (LLM_PROVIDER=none hoặc thiếu model).",
            )

        assert self.client is not None  # Cho type checker
        chunks = split_into_chunks(text, self.config.max_chars_per_chunk)
        corrected_parts: list[str] = []
        corrections: list[Correction] = []
        errors: list[str] = []
        rejected = 0
        accepted = 0

        for index, chunk in enumerate(chunks, start=1):
            if not chunk.strip():
                corrected_parts.append(chunk)
                continue
            user_prompt = USER_PROMPT_TEMPLATE.format(
                context=context.strip() or "（指定なし）", text=chunk
            )
            try:
                raw = self.client.complete(SYSTEM_PROMPT, user_prompt)
                payload = parse_llm_json(raw)
                candidate = str(payload.get("corrected_text", "")).strip("\r\n")
                if not candidate:
                    raise ValueError("LLM trả về corrected_text rỗng.")

                ratio = compute_edit_ratio(chunk, candidate)
                if ratio > self.config.max_edit_ratio:
                    # Guard chống hallucination: sửa quá nhiều => nhiều khả năng LLM đã
                    # viết lại câu chứ không chỉ sửa lỗi OCR. Giữ nguyên bản gốc.
                    rejected += 1
                    logger.warning(
                        "Chunk %d/%d bị từ chối: edit_ratio=%.3f > %.3f",
                        index, len(chunks), ratio, self.config.max_edit_ratio,
                    )
                    corrected_parts.append(chunk)
                    continue

                corrected_parts.append(candidate)
                corrections.extend(self._parse_corrections(payload))
                accepted += 1
            except Exception as exc:  # Lỗi mạng, rate limit, JSON hỏng... => fail-safe
                logger.error("Chunk %d/%d lỗi post-correction: %s", index, len(chunks), exc)
                errors.append(f"chunk {index}: {exc}")
                corrected_parts.append(chunk)

        corrected_text = "\n".join(corrected_parts)
        message = (
            f"Đã xử lý {len(chunks)} đoạn: {accepted} áp dụng, {rejected} bị guard từ chối, "
            f"{len(errors)} lỗi."
        )
        return PostCorrectionResult(
            original_text=text,
            corrected_text=corrected_text,
            corrections=corrections,
            applied=accepted > 0,
            edit_ratio=compute_edit_ratio(text, corrected_text),
            provider=self.config.provider,
            model=self.config.model,
            rejected_chunks=rejected,
            errors=errors,
            message=message,
        )

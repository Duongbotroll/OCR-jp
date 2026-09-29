"""Giao diện Web Demo (Streamlit) cho hệ thống OCR chữ viết tay tiếng Nhật.

Chạy:
    streamlit run app.py

Luồng: Upload (ảnh/PDF/TIFF) -> Tầng 1 Preprocess -> Tầng 2 VLM OCR
       -> Tầng 3 LLM Post-Correction -> Hiển thị diff, đánh giá CER, xuất JSON/TXT.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from typing import Any

import numpy as np
import streamlit as st

from config import (
    DEFAULT_LLM_MODELS,
    DEFAULT_MODEL_IDS,
    SUPPORTED_LLM_PROVIDERS,
    SUPPORTED_OCR_BACKENDS,
    SUPPORTED_OCR_MODES,
    SUPPORTED_TEXT_DIRECTIONS,
    OCRConfig,
    PostCorrectionConfig,
    PreprocessConfig,
    Settings,
    load_settings,
)
from src import __version__
from src.ocr_engine import DEFAULT_OCR_PROMPT, OCREngine
from src.postprocessor import JapanesePostCorrector, extract_char_changes, render_diff_html
from src.preprocessor import ImagePreprocessor
from src.utils import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    Timer,
    character_error_rate,
    load_document_pages,
    setup_logger,
)

logger = setup_logger("app")

st.set_page_config(
    page_title="Japanese Handwritten OCR",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Nhãn thân thiện cho từng bước tiền xử lý (hiển thị trong tab Preprocessing)
STEP_LABELS: dict[str, str] = {
    "01_input": "Input (resized)",
    "02_denoised": "Denoised (NLM)",
    "03_contrast": "Contrast (CLAHE)",
    "04_deskewed": "Deskewed",
    "05_cropped": "Border cropped",
    "06_binary": "Binary (layout map)",
}


# ---------------------------------------------------------------------------
# Streamlit Secrets -> biến môi trường
# ---------------------------------------------------------------------------
def export_streamlit_secrets_to_env() -> None:
    """Đưa các secret cấp gốc (Streamlit Cloud / .streamlit/secrets.toml) vào os.environ.

    Nhờ vậy ``config.load_settings()`` đọc được cấu hình giống hệt khi chạy local với
    file ``.env``. Biến môi trường đã có sẵn trong shell được ưu tiên (không ghi đè).
    Khi chạy local mà không có secrets.toml, hàm im lặng bỏ qua.
    """
    try:
        items = list(st.secrets.items())
    except Exception:  # Không có file secrets -> dùng .env như bình thường
        return
    for key, value in items:
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(key, str(value).lower() if isinstance(value, bool)
                                  else str(value))


# ---------------------------------------------------------------------------
# Tải model (cache giữa các lần rerun của Streamlit)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading OCR model (lần đầu có thể tải vài GB weights)…")
def get_ocr_engine(config_items: tuple[tuple[str, Any], ...]) -> OCREngine:
    """Tạo OCREngine và giữ trong bộ nhớ.

    Tham số là tuple các cặp (key, value) nguyên thủy để Streamlit băm (hash) ổn định;
    khi người dùng đổi model/thiết bị trong sidebar, một engine mới sẽ được tạo.
    """
    return OCREngine(OCRConfig(**dict(config_items)))


def _config_key(config: OCRConfig) -> tuple[tuple[str, Any], ...]:
    """Chuyển OCRConfig thành khóa cache có thể hash."""
    return tuple(sorted(asdict(config).items()))


# ---------------------------------------------------------------------------
# Sidebar cấu hình
# ---------------------------------------------------------------------------
def render_sidebar(base: Settings) -> tuple[Settings, str, str]:
    """Vẽ sidebar và trả về (settings đã chỉnh, prompt OCR, ngữ cảnh tài liệu)."""
    st.sidebar.title("⚙️ Pipeline Settings")
    st.sidebar.caption(f"v{__version__}")

    with st.sidebar.expander("① Pre-processing & Layout", expanded=False):
        pre = base.preprocess
        denoise = st.checkbox("Denoise (Non-Local Means)", value=pre.denoise)
        clahe = st.checkbox("Contrast enhancement (CLAHE)", value=pre.clahe)
        deskew = st.checkbox("Auto deskew", value=pre.deskew)
        max_skew = st.slider("Max skew angle (°)", 1.0, 20.0, float(pre.max_skew_angle), 0.5)
        crop = st.checkbox("Crop empty borders", value=pre.crop_borders)
        binarize = st.checkbox("Feed binarized image to OCR", value=pre.binarize_for_ocr)
        layout = st.checkbox("Layout analysis", value=pre.layout_analysis)
        direction = st.selectbox(
            "Text direction",
            SUPPORTED_TEXT_DIRECTIONS,
            index=SUPPORTED_TEXT_DIRECTIONS.index(pre.text_direction),
            help="auto = tự phát hiện 横書き (ngang) / 縦書き (dọc)",
        )
        max_side = st.select_slider(
            "Max image side (px)", options=[1024, 1536, 2048, 2560, 3072],
            value=pre.max_side if pre.max_side in (1024, 1536, 2048, 2560, 3072) else 2048,
        )

    with st.sidebar.expander("② OCR Engine (VLM)", expanded=True):
        ocr = base.ocr
        backend = st.selectbox(
            "Backend", SUPPORTED_OCR_BACKENDS,
            index=SUPPORTED_OCR_BACKENDS.index(ocr.backend),
        )
        is_api = backend == "openai-compatible"
        default_model = ocr.model_id if backend == ocr.backend else DEFAULT_MODEL_IDS[backend]
        model_id = ocr.model_id
        api_base_url, api_model, api_key = ocr.api_base_url, ocr.api_model, ocr.api_key
        if is_api:
            api_base_url = st.text_input("API base URL", value=ocr.api_base_url)
            api_model = st.text_input("Served model name", value=ocr.api_model)
            api_key = st.text_input("API key (optional)", value=ocr.api_key, type="password")
        else:
            # key phụ thuộc backend để ô nhập tự reset về model mặc định khi đổi backend
            model_id = st.text_input("Hugging Face model ID", value=default_model,
                                     key=f"model_id_{backend}")
        device = st.selectbox("Device", ["auto", "cuda", "mps", "cpu"],
                              index=["auto", "cuda", "mps", "cpu"].index(ocr.device)
                              if ocr.device in ("auto", "cuda", "mps", "cpu") else 0,
                              disabled=is_api)
        mode = st.radio("OCR mode", SUPPORTED_OCR_MODES,
                        index=SUPPORTED_OCR_MODES.index(ocr.mode), horizontal=True,
                        help="page = cả trang (khuyến nghị) | block = từng khối layout")
        max_new_tokens = st.number_input("Max new tokens", 128, 8192, ocr.max_new_tokens, 128)

    with st.sidebar.expander("③ LLM Post-Correction", expanded=True):
        post = base.post
        provider = st.selectbox(
            "Provider", SUPPORTED_LLM_PROVIDERS,
            index=SUPPORTED_LLM_PROVIDERS.index(post.provider),
        )
        default_llm = post.model if provider == post.provider else DEFAULT_LLM_MODELS[provider]
        llm_model = st.text_input("LLM model", value=default_llm, key=f"llm_model_{provider}",
                                  disabled=provider == "none")
        llm_key = post.api_key if provider == post.provider else ""
        llm_key = st.text_input("API key", value=llm_key, type="password",
                                disabled=provider == "none",
                                help="Để trống để dùng biến môi trường trong .env")
        max_edit_ratio = st.slider(
            "Hallucination guard: max edit ratio", 0.05, 0.60, float(post.max_edit_ratio), 0.05,
            help="Nếu LLM thay đổi quá tỉ lệ này, bản sửa bị từ chối và giữ nguyên OCR gốc.",
        )
        context = st.text_input(
            "Document context (optional)", value="",
            placeholder="例: 医療問診票 / 請求書 / 申込書 / 手紙",
        )

    with st.sidebar.expander("Advanced: OCR prompt", expanded=False):
        prompt = st.text_area("Prompt", value=DEFAULT_OCR_PROMPT, height=260)

    settings = Settings(
        preprocess=PreprocessConfig(
            max_side=int(max_side), denoise=denoise, denoise_strength=pre.denoise_strength,
            clahe=clahe, deskew=deskew, max_skew_angle=float(max_skew), crop_borders=crop,
            binarize_for_ocr=binarize, layout_analysis=layout,
            min_block_area=pre.min_block_area, text_direction=direction,
        ),
        ocr=OCRConfig(
            backend=backend, model_id=model_id if not is_api else "", device=device,
            dtype=ocr.dtype, mode=mode, max_new_tokens=int(max_new_tokens),
            repetition_penalty=ocr.repetition_penalty,
            trust_remote_code=ocr.trust_remote_code, cache_dir=ocr.cache_dir,
            qwen_max_pixels=ocr.qwen_max_pixels, api_base_url=api_base_url,
            api_key=api_key, api_model=api_model, api_timeout=ocr.api_timeout,
        ),
        post=PostCorrectionConfig(
            provider=provider, model=llm_model if provider != "none" else "",
            api_key=llm_key, base_url=post.base_url if provider == "openai" else "",
            temperature=post.temperature, max_tokens=post.max_tokens, timeout=post.timeout,
            max_edit_ratio=float(max_edit_ratio),
            max_chars_per_chunk=post.max_chars_per_chunk,
        ),
        log_level=base.log_level,
        output_dir=base.output_dir,
        pdf_dpi=base.pdf_dpi,
        max_pdf_pages=base.max_pdf_pages,
    )
    return settings, prompt, context


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def process_document(
    pages: list[np.ndarray], settings: Settings, prompt: str, context: str
) -> list[dict[str, Any]]:
    """Chạy đủ 3 tầng xử lý cho từng trang và trả về danh sách kết quả."""
    preprocessor = ImagePreprocessor(settings.preprocess)
    engine = get_ocr_engine(_config_key(settings.ocr))
    corrector = JapanesePostCorrector(settings.post)

    results: list[dict[str, Any]] = []
    progress = st.progress(0.0, text="Starting pipeline…")
    for page_number, page in enumerate(pages, start=1):
        progress.progress((page_number - 1) / len(pages),
                          text=f"Page {page_number}/{len(pages)}: pre-processing…")
        with Timer() as pre_timer:
            pre = preprocessor.process(page)

        progress.progress((page_number - 0.66) / len(pages),
                          text=f"Page {page_number}/{len(pages)}: VLM OCR…")
        ocr = engine.run(pre, prompt=prompt)

        progress.progress((page_number - 0.33) / len(pages),
                          text=f"Page {page_number}/{len(pages)}: LLM post-correction…")
        with Timer() as post_timer:
            post = corrector.correct(ocr.text, context=context)

        results.append(
            {
                "page": page_number,
                "pre": pre,
                "ocr": ocr,
                "post": post,
                "overlay": preprocessor.draw_blocks(pre.image, pre.blocks),
                "timings": {
                    "preprocess_s": round(pre_timer.elapsed, 3),
                    "ocr_s": ocr.elapsed_seconds,
                    "post_correction_s": round(post_timer.elapsed, 3),
                },
            }
        )
    progress.progress(1.0, text="Done ✅")
    return results


def build_export(
    results: list[dict[str, Any]], settings: Settings, filename: str
) -> dict[str, Any]:
    """Tạo cấu trúc JSON chuẩn hóa để tích hợp với hệ thống khác (RPA, DB, API)."""
    return {
        "schema_version": "1.0",
        "app_version": __version__,
        "source_file": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": {
            "ocr_backend": settings.ocr.backend,
            "ocr_model": settings.ocr.api_model or settings.ocr.model_id,
            "ocr_mode": settings.ocr.mode,
            "llm_provider": settings.post.provider,
            "llm_model": settings.post.model,
        },
        "pages": [
            {
                "page": item["page"],
                "orientation": item["pre"].orientation,
                "skew_angle": item["pre"].skew_angle,
                "blocks": [asdict(block) for block in item["pre"].blocks],
                "ocr_text": item["ocr"].text,
                "corrected_text": item["post"].corrected_text,
                "llm_applied": item["post"].applied,
                "edit_ratio": item["post"].edit_ratio,
                "llm_corrections": [asdict(c) for c in item["post"].corrections],
                "verified_changes": extract_char_changes(
                    item["ocr"].text, item["post"].corrected_text
                ),
                "timings": item["timings"],
            }
            for item in results
        ],
    }


# ---------------------------------------------------------------------------
# Hiển thị kết quả
# ---------------------------------------------------------------------------
def render_page_result(item: dict[str, Any], run_id: int) -> None:
    """Hiển thị kết quả của một trang dưới dạng các tab."""
    pre, ocr, post = item["pre"], item["ocr"], item["post"]
    page = item["page"]
    changes = extract_char_changes(ocr.text, post.corrected_text)

    cols = st.columns(5)
    cols[0].metric("Direction", "縦書き (vertical)" if pre.orientation == "vertical"
                   else "横書き (horizontal)")
    cols[1].metric("Skew fixed", f"{pre.skew_angle:.2f}°")
    cols[2].metric("Text blocks", len(pre.blocks))
    cols[3].metric("OCR time", f"{ocr.elapsed_seconds:.1f}s")
    cols[4].metric("LLM edits", len(changes))

    tab_pre, tab_ocr, tab_post, tab_eval = st.tabs(
        ["🖼️ Pre-processing", "🔍 Raw OCR", "✍️ Post-correction", "📊 Evaluation"]
    )

    with tab_pre:
        step_items = list(pre.steps.items())
        for start in range(0, len(step_items), 3):
            row = st.columns(3)
            for col, (key, image) in zip(row, step_items[start : start + 3]):
                col.image(image, caption=STEP_LABELS.get(key, key), channels="BGR")
        st.image(item["overlay"], caption="Layout analysis & reading order", channels="BGR")

    with tab_ocr:
        st.caption(f"Backend: `{ocr.backend}` · Model: `{ocr.model_id}` · Mode: `{ocr.mode}`")
        st.text_area("Raw OCR output", ocr.text, height=320, key=f"raw_{run_id}_{page}")
        if ocr.block_texts:
            st.markdown("**Per-block output (reading order):**")
            for index, text in enumerate(ocr.block_texts, start=1):
                st.markdown(f"`#{index}` {text if text else '_(empty)_'}")

    with tab_post:
        if post.applied:
            st.success(post.message)
        else:
            st.info(post.message or "Post-correction not applied.")
        for error in post.errors:
            st.warning(error)
        left, right = st.columns(2)
        left.text_area("Before (OCR)", post.original_text, height=260,
                       key=f"before_{run_id}_{page}")
        right.text_area("After (LLM corrected)", post.corrected_text, height=260,
                        key=f"after_{run_id}_{page}")
        st.markdown("**Character-level diff**")
        st.markdown(render_diff_html(post.original_text, post.corrected_text),
                    unsafe_allow_html=True)
        if changes:
            st.markdown("**Verified changes (computed by diff):**")
            st.dataframe(changes, hide_index=True)
        if post.corrections:
            st.markdown("**LLM rationale (self-reported):**")
            st.dataframe([asdict(c) for c in post.corrections], hide_index=True)

    with tab_eval:
        st.caption("Dán ground truth để tính Character Error Rate (CER) trước và sau LLM.")
        truth = st.text_area("Ground truth", height=160, key=f"gt_{run_id}_{page}")
        if truth.strip():
            cer_raw = character_error_rate(truth, ocr.text)
            cer_post = character_error_rate(truth, post.corrected_text)
            c1, c2, c3 = st.columns(3)
            c1.metric("CER · raw OCR", f"{cer_raw:.2%}")
            c2.metric("CER · after LLM", f"{cer_post:.2%}", delta=f"{cer_post - cer_raw:+.2%}",
                      delta_color="inverse")
            c3.metric("Accuracy · after LLM", f"{max(0.0, 1 - cer_post):.2%}")
        st.json(item["timings"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Điểm vào của ứng dụng Streamlit."""
    export_streamlit_secrets_to_env()
    base_settings = load_settings()
    settings, prompt, context = render_sidebar(base_settings)

    st.title("📝 Japanese Handwritten Document OCR")
    st.markdown(
        "**3-Stage Pipeline:** OpenCV pre-processing → SOTA Vision-Language OCR → "
        "LLM contextual post-correction · 手書き日本語文書のOCR・構造化"
    )

    extensions = sorted(ext.lstrip(".") for ext in SUPPORTED_DOCUMENT_EXTENSIONS)
    uploaded = st.file_uploader("Upload a handwritten document (image / PDF / TIFF)",
                                type=extensions)
    if uploaded is None:
        st.info(
            "⬆️ Upload a scan or phone photo of a handwritten Japanese document "
            "(forms, letters, notes, questionnaires). Configure the pipeline in the sidebar."
        )
        return

    try:
        pages = load_document_pages(uploaded.getvalue(), uploaded.name,
                                    pdf_dpi=settings.pdf_dpi, max_pages=settings.max_pdf_pages)
    except Exception as exc:
        st.error(f"Cannot read the document: {exc}")
        return

    st.caption(f"Loaded **{len(pages)}** page(s) from `{uploaded.name}`")
    with st.expander("Preview", expanded=False):
        preview_cols = st.columns(min(4, len(pages)))
        for index, page in enumerate(pages[:4]):
            preview_cols[index].image(page, caption=f"Page {index + 1}", channels="BGR")

    if st.button("🚀 Run OCR pipeline", type="primary"):
        try:
            results = process_document(pages, settings, prompt, context)
        except Exception as exc:
            logger.exception("Pipeline failed")
            st.error(
                f"Pipeline failed: {exc}\n\n"
                "Gợi ý: kiểm tra model ID, dung lượng VRAM/RAM, phiên bản transformers, "
                "hoặc cấu hình API trong file .env."
            )
            return
        st.session_state["results"] = results
        st.session_state["result_file"] = uploaded.name
        st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1
        st.session_state["settings"] = settings

    results = st.session_state.get("results")
    if not results or st.session_state.get("result_file") != uploaded.name:
        return

    run_id = st.session_state.get("run_id", 0)
    for item in results:
        st.divider()
        st.subheader(f"Page {item['page']}")
        render_page_result(item, run_id)

    st.divider()
    st.subheader("📦 Export")
    export = build_export(results, st.session_state["settings"], uploaded.name)
    stem = uploaded.name.rsplit(".", 1)[0]
    full_text = "\n\n".join(
        f"===== Page {item['page']} =====\n{item['post'].corrected_text}" for item in results
    )
    col_json, col_txt = st.columns(2)
    col_json.download_button(
        "⬇️ Download JSON (structured)",
        data=json.dumps(export, ensure_ascii=False, indent=2),
        file_name=f"{stem}_ocr.json", mime="application/json",
    )
    col_txt.download_button(
        "⬇️ Download TXT (corrected text)",
        data=full_text, file_name=f"{stem}_ocr.txt", mime="text/plain",
    )


if __name__ == "__main__":
    main()

<div align="center">

# 📝 Japanese Handwritten Document OCR & Parsing System

**手書き日本語文書 OCR・構造化システム**

*SOTA Vision-Language Models + LLM Contextual Post-Correction for Japanese Digital Transformation (DX)*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.51%2B-FFD21E)](https://huggingface.co/docs/transformers)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![Code style: PEP8](https://img.shields.io/badge/code%20style-PEP8-000000)](https://peps.python.org/pep-0008/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](CHANGELOG.md)

[Features](#-key-features) · [Architecture](#-architecture) · [Quickstart](#-quickstart) · [Configuration](#%EF%B8%8F-configuration) · [Docs](ARCHITECTURE.md) · [Tiếng Việt](#-tóm-tắt-tiếng-việt)

</div>

---

## 🎯 Why this project?

Japan still runs on paper. Application forms (申込書), medical questionnaires (問診票), invoices (請求書), delivery slips (伝票), and handwritten letters are processed manually across government offices, hospitals, insurers, and logistics companies. Digitizing them is a core DX (デジタルトランスフォーメーション) initiative — and **handwritten Japanese is one of the hardest OCR problems in production**:

| Challenge | Why it is hard |
|---|---|
| **Huge character set** | 2,136 常用漢字 + hiragana + katakana + Latin + full-width symbols, all mixed in one line |
| **Confusable glyphs** | 未/末, 土/士, シ/ツ, ソ/ン, カ/力, ロ/口, ニ/二, ー/一 — often indistinguishable without context |
| **Two writing directions** | 横書き (horizontal, left→right) and 縦書き (vertical, right→left columns) |
| **No word spacing** | Word boundaries cannot be used to recover errors |
| **Real-world capture** | Phone photos with shadows, skew, yellowed paper, stamps (印鑑) overlapping text |

This repository implements a **production-oriented 3-stage pipeline** that combines classic computer vision, state-of-the-art Vision-Language Models (VLMs), and guarded LLM post-correction.

## ✨ Key Features

- 🧹 **Robust pre-processing** — NLM denoising, CLAHE, projection-profile deskew (works for both writing directions), border cropping.
- 🧭 **Japanese-aware layout analysis** — automatic 横書き / 縦書き detection and **correct Japanese reading order** (right-to-left columns for vertical text).
- 🤖 **Pluggable SOTA OCR backends** — Qwen2.5-VL, GLM-OCR, GOT-OCR 2.0, or any VLM served behind an OpenAI-compatible API (vLLM / SGLang). Swap models with one env variable.
- ✍️ **LLM contextual post-correction** — fixes confusable Kanji/Kana using sentence context, with structured JSON output and per-edit rationale.
- 🛡️ **Hallucination guard** — edit-ratio threshold rejects corrections that rewrite rather than repair; diff-verified change log for human-in-the-loop review.
- 📊 **Built-in evaluation** — Character Error Rate (CER) with NFKC normalization, before vs. after LLM correction.
- 📄 **Multi-format input** — PNG/JPG/WebP/BMP, multi-page TIFF (office scanners), and PDF (via PyMuPDF, no Poppler needed).
- 🖥️ **Interactive Streamlit demo** — visualize every pre-processing step, block overlay, character-level diff, and export JSON/TXT.
- 🐳 **Deploy-ready** — Dockerfile (CPU/GPU), non-root container, healthcheck, 12-factor configuration.
- 🔒 **Privacy by design** — can run 100% on-premise (local VLM + local LLM through vLLM/Ollama), aligned with Japan's APPI (個人情報保護法).

## 🏗 Architecture

```text
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │                        INPUT: Scan / Phone photo / PDF / TIFF                 │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │ load_document_pages()  (EXIF fix, PDF render)
                                         ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1 ─ PRE-PROCESSING & LAYOUT ANALYSIS                 (src/preprocessor) │
 │                                                                               │
 │  Resize ─► Gray ─► NLM Denoise ─► CLAHE ─► Adaptive Binarize                  │
 │                                               │                               │
 │            ┌──────────────────────────────────┘                               │
 │            ▼                                                                  │
 │  Projection-profile Deskew ─► Border Crop ─► Direction detect (横 / 縦)       │
 │                                               │                               │
 │                                               ▼                               │
 │                      Morphological Layout Analysis ─► Japanese Reading Order  │
 │                      (横: top→bottom, left→right | 縦: right→left, top→bottom)│
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │ PreprocessResult(image, blocks, orientation)
                                         ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2 ─ VISION-LANGUAGE OCR ENGINE                         (src/ocr_engine) │
 │                                                                               │
 │   OCREngine ──(mode = page | block)──► BaseOCRBackend  (Strategy + Factory)   │
 │                                          ├─ QwenVLBackend        Qwen2.5-VL   │
 │                                          ├─ GLMOCRBackend        GLM-OCR      │
 │                                          ├─ GOTOCR2Backend       GOT-OCR 2.0  │
 │                                          └─ OpenAICompatible...  vLLM / SGLang│
 │   Greedy decoding · repetition penalty · output sanitization                  │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │ OCRResult(text, block_texts, latency)
                                         ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 3 ─ LLM CONTEXTUAL POST-CORRECTION                  (src/postprocessor) │
 │                                                                               │
 │  Chunking ─► Strict JA prompt ─► LLM (OpenAI / Claude / local) ─► JSON parse  │
 │                                                                  │            │
 │                     ┌────────────────────────────────────────────┘            │
 │                     ▼                                                         │
 │        Hallucination Guard: edit_ratio(original, corrected) <= threshold ?    │
 │              ├─ YES ─► accept correction                                      │
 │              └─ NO  ─► keep original OCR text (fail-safe)                     │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ OUTPUT: corrected text · verified diff · LLM rationale · CER · JSON / TXT     │
 │         Streamlit UI (app.py)  ──►  downstream RPA / DB / search index        │
 └───────────────────────────────────────────────────────────────────────────────┘
```

A deep dive into every design decision, supported models, and the benchmark methodology is in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## 🤖 Supported OCR Backends

| Backend key | Default model | Type | Best for | Notes |
|---|---|---|---|---|
| `qwen2.5-vl` | `Qwen/Qwen2.5-VL-7B-Instruct` | General VLM | Best overall Japanese understanding, mixed layouts | 3B variant for 8–12 GB GPUs; 72B for maximum accuracy |
| `glm-ocr` | `zai-org/GLM-OCR` | OCR-specialized VLM | Fast, lightweight document OCR | Check the model card for the minimum `transformers` version |
| `got-ocr2` | `stepfun-ai/GOT-OCR-2.0-hf` | End-to-end OCR | Very fast, small footprint | Trained mostly on EN/ZH — pair with LLM post-correction for Japanese |
| `openai-compatible` | *(your served model)* | Remote API | Production GPU servers (vLLM, SGLang, LMDeploy) | Keeps the web app thin and stateless |

> Model availability and licenses change quickly. Always check each model card on Hugging Face before production use.

## 🚀 Quickstart

### Prerequisites

- Python **3.10+**
- For local VLM inference: NVIDIA GPU recommended (≈ 8 GB VRAM for 3B models, ≈ 16–24 GB for 7B in bf16). CPU works for testing but is slow.
- An API key for post-correction (OpenAI or Anthropic) — **or** a local OpenAI-compatible server (Ollama / vLLM), **or** set `LLM_PROVIDER=none`.

### Option A — One-command setup (recommended)

```bash
git clone https://github.com/<your-username>/japanese-handwritten-ocr.git
cd japanese-handwritten-ocr

# CPU / Apple Silicon
./run.sh

# NVIDIA GPU (install CUDA build of PyTorch first)
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./run.sh
```

`run.sh` creates a virtualenv, installs dependencies, creates `.env` from `.env.example`, and launches the app at **http://localhost:8501**.

### Option B — Docker

```bash
cp .env.example .env         # fill in your keys

# CPU
docker build -t jp-handwritten-ocr .
docker run --rm -p 8501:8501 --env-file .env jp-handwritten-ocr

# GPU
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 -t jp-handwritten-ocr:gpu .
docker run --rm --gpus all -p 8501:8501 --env-file .env \
  -v hf_cache:/app/.cache/huggingface jp-handwritten-ocr:gpu
```

### Option C — Manual

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # core only: OCR via API (Gemini / OpenAI / vLLM)
pip install -r requirements-gpu.txt    # + PyTorch/Transformers for local VLM inference
cp .env.example .env
streamlit run app.py
```

> **Two dependency sets:** `requirements.txt` is lightweight (no PyTorch) and is what Streamlit Community Cloud installs. `requirements-gpu.txt` adds PyTorch + Transformers for running Qwen2.5-VL / GLM-OCR / GOT-OCR2 locally. With `run.sh`, use `INSTALL_MODE=lite ./run.sh` for the API-only setup.

### Option D — Free cloud demo (Streamlit Community Cloud + Gemini API, no GPU)

1. Push this repo to GitHub (public).
2. Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey).
3. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → select the repo, branch `main`, main file `app.py`; under **Advanced settings** choose Python 3.11 and paste the content of [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) (with your key) into **Secrets**.
4. Deploy — the app installs only the lightweight `requirements.txt`, and both OCR (Stage 2) and post-correction (Stage 3) run through Gemini's OpenAI-compatible endpoint.

> ⚠️ Do not upload real personal documents to a public demo or to a free-tier API. Use your own handwriting samples.

### Option E — Production split (VLM server + thin client)

```bash
# On the GPU server
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000

# In .env of the web app
OCR_BACKEND=openai-compatible
OCR_API_BASE_URL=http://<gpu-server>:8000/v1
OCR_API_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

## 🧪 Programmatic Usage

```python
from config import load_settings
from src.utils import load_image
from src.preprocessor import ImagePreprocessor
from src.ocr_engine import OCREngine
from src.postprocessor import JapanesePostCorrector

settings = load_settings()

page = load_image("samples/handwritten_form.jpg")
pre = ImagePreprocessor(settings.preprocess).process(page)
ocr = OCREngine(settings.ocr).run(pre)
post = JapanesePostCorrector(settings.post).correct(ocr.text, context="医療問診票")

print("Direction :", pre.orientation, "| skew:", pre.skew_angle)
print("Raw OCR   :", ocr.text)
print("Corrected :", post.corrected_text)
for c in post.corrections:
    print(f"  {c.original} → {c.corrected}  ({c.reason})")
```

## ⚙️ Configuration

All settings live in `.env` (see [`.env.example`](.env.example)) and are validated in [`config.py`](config.py). Run `python config.py` to print the effective configuration.

| Variable | Default | Description |
|---|---|---|
| `OCR_BACKEND` | `qwen2.5-vl` | `qwen2.5-vl` · `glm-ocr` · `got-ocr2` · `openai-compatible` |
| `OCR_MODEL_ID` | *(backend default)* | Any compatible Hugging Face model ID |
| `OCR_DEVICE` | `auto` | `auto` → CUDA › MPS › CPU |
| `OCR_MODE` | `page` | `page` (whole page, recommended) or `block` (per layout block) |
| `PREPROCESS_TEXT_DIRECTION` | `auto` | Force `horizontal` / `vertical` if detection is wrong |
| `LLM_PROVIDER` | `openai` | `openai` · `anthropic` · `none` |
| `OPENAI_BASE_URL` | *(empty)* | Point to Ollama / vLLM for fully local post-correction |
| `LLM_MAX_EDIT_RATIO` | `0.25` | Hallucination guard threshold (fraction of characters changed) |

## 📁 Project Structure

```text
japanese-handwritten-ocr/
├── .gitignore               # Blocks weights, .env, caches, venv
├── .env.example             # Configuration template (incl. no-GPU Gemini preset)
├── .dockerignore            # Keeps secrets & venv out of the Docker image
├── .streamlit/
│   └── secrets.toml.example # Secrets template for Streamlit Community Cloud
├── LICENSE                  # MIT
├── README.md                # You are here
├── CONTRIBUTING.md          # Contribution workflow & coding standards
├── CHANGELOG.md             # Release notes
├── ARCHITECTURE.md          # Technical deep dive & benchmark methodology
├── requirements.txt         # Core dependencies (no PyTorch; used by Streamlit Cloud)
├── requirements-gpu.txt     # + PyTorch/Transformers for local VLM inference
├── Dockerfile               # CPU/GPU container
├── config.py                # Typed, validated 12-factor configuration
├── src/
│   ├── __init__.py
│   ├── utils.py             # Image/PDF I/O, logging, timer, CER metric
│   ├── preprocessor.py      # Stage 1: OpenCV pre-processing & layout analysis
│   ├── ocr_engine.py        # Stage 2: VLM backends (Strategy + Factory)
│   └── postprocessor.py     # Stage 3: guarded LLM post-correction
├── app.py                   # Streamlit web demo
└── run.sh                   # Install & launch automation
```

## 📤 Output Schema (JSON)

```json
{
  "schema_version": "1.0",
  "source_file": "form.pdf",
  "pipeline": { "ocr_backend": "qwen2.5-vl", "ocr_model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "ocr_mode": "page", "llm_provider": "openai", "llm_model": "gpt-4o-mini" },
  "pages": [
    {
      "page": 1,
      "orientation": "horizontal",
      "skew_angle": -1.3,
      "blocks": [{ "x": 40, "y": 62, "w": 810, "h": 58, "order": 0 }],
      "ocr_text": "…",
      "corrected_text": "…",
      "llm_applied": true,
      "edit_ratio": 0.021,
      "llm_corrections": [{ "original": "末", "corrected": "未", "reason": "…" }],
      "verified_changes": [{ "operation": "replace", "position": 12, "original": "末", "corrected": "未" }],
      "timings": { "preprocess_s": 0.41, "ocr_s": 6.8, "post_correction_s": 1.9 }
    }
  ]
}
```

## ⚠️ Known Limitations

- Direction detection and layout analysis are heuristic (no learned layout model yet) — complex forms with tables may need `OCR_MODE=page`.
- Heavily cursive / historical script (くずし字) is out of scope for the default models.
- LLM post-correction cannot recover characters that are genuinely ambiguous; it is designed to be conservative and keep the original when unsure.
- Accuracy numbers depend strongly on your documents — measure with your own labeled samples (see [ARCHITECTURE.md § Benchmark](ARCHITECTURE.md#7-benchmark-methodology)).

## 🗺 Roadmap

- [ ] Learned layout detection (table / form-field / stamp detection)
- [ ] Key-value extraction to typed schemas (e.g., 氏名, 住所, 生年月日) with Pydantic
- [ ] FastAPI REST service + async batch queue
- [ ] Evaluation CLI over labeled datasets with per-character-class error reports
- [ ] LoRA fine-tuning recipe for domain-specific handwriting

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md).

## 📜 License

Released under the [MIT License](LICENSE). Third-party model weights are subject to their own licenses.

## 🙏 Acknowledgements

Qwen team (Alibaba), Z.ai (GLM-OCR), StepFun (GOT-OCR 2.0), Hugging Face Transformers, OpenCV, and Streamlit.

---

## 🇻🇳 Tóm tắt tiếng Việt

**Japanese Handwritten Document OCR** là hệ thống nhận dạng và bóc tách tài liệu viết tay tiếng Nhật, phục vụ các dự án Chuyển đổi số (DX) tại thị trường Nhật Bản — nơi hồ sơ giấy như đơn đăng ký, phiếu khám bệnh, hóa đơn vẫn được xử lý thủ công.

**Kiến trúc 3 tầng:**

1. **Tiền xử lý & phân tích bố cục (OpenCV):** khử nhiễu, tăng tương phản CLAHE, chỉnh nghiêng bằng projection profile, cắt viền, tự nhận diện chữ viết ngang (横書き) hay dọc (縦書き) và sắp xếp đúng thứ tự đọc tiếng Nhật.
2. **OCR bằng Vision-Language Model SOTA:** hỗ trợ Qwen2.5-VL, GLM-OCR, GOT-OCR 2.0 hoặc bất kỳ VLM nào phục vụ qua API chuẩn OpenAI (vLLM). Đổi model chỉ bằng một biến môi trường nhờ Strategy + Factory Pattern.
3. **LLM hậu xử lý theo ngữ cảnh:** sửa các lỗi nhầm chữ có hình dạng gần giống (未/末, シ/ツ, カ/力...) với prompt nghiêm ngặt, đầu ra JSON có lý do từng chỗ sửa, và **cơ chế chống hallucination** (từ chối bản sửa nếu thay đổi quá ngưỡng).

**Chạy nhanh:** `./run.sh` (hoặc Docker) rồi mở `http://localhost:8501`. Không có GPU? Dùng preset Gemini (`INSTALL_MODE=lite ./run.sh`) hoặc deploy miễn phí lên Streamlit Community Cloud theo Option D. Giao diện Streamlit hiển thị từng bước tiền xử lý, khung layout, diff từng ký tự, tính CER khi có ground truth và xuất JSON/TXT.

**Điểm nổi bật khi phỏng vấn AI Engineer:** thiết kế module hóa, cấu hình 12-factor, fail-safe khi LLM lỗi, đánh giá bằng CER, có thể chạy hoàn toàn on-premise để tuân thủ luật bảo vệ thông tin cá nhân APPI của Nhật.

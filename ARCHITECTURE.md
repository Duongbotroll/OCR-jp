# 🏛 Architecture — Japanese Handwritten Document OCR & Parsing System

> Technical deep dive for engineers and reviewers. For setup instructions see [README.md](README.md).
> Tóm tắt tiếng Việt ở [cuối tài liệu](#-tóm-tắt-tiếng-việt).

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Design Goals & Non-Goals](#2-design-goals--non-goals)
3. [System Overview](#3-system-overview)
4. [Stage 1 — Pre-processing & Layout Analysis](#4-stage-1--pre-processing--layout-analysis)
5. [Stage 2 — Vision-Language OCR Engine](#5-stage-2--vision-language-ocr-engine)
6. [Stage 3 — LLM Contextual Post-Correction](#6-stage-3--llm-contextual-post-correction)
7. [Benchmark Methodology](#7-benchmark-methodology)
8. [Performance & Scaling](#8-performance--scaling)
9. [Security, Privacy & Compliance](#9-security-privacy--compliance)
10. [Failure Modes & Mitigations](#10-failure-modes--mitigations)
11. [Extensibility Guide](#11-extensibility-guide)
12. [Architecture Decision Records](#12-architecture-decision-records)

---

## 1. Problem Statement

Input: a scanned image, phone photo, multi-page TIFF, or PDF of a **handwritten Japanese document**.
Output: **faithful machine-readable text** in correct reading order, plus metadata (layout blocks, skew, direction, correction log) suitable for downstream RPA, databases, or search.

"Faithful" is the key word. In DX projects for Japanese insurers, hospitals, and municipalities, the cost of a *confidently wrong* character (a changed amount, date, or name) is far higher than the cost of a flagged uncertainty. Every design decision below trades a little raw fluency for **auditability and conservatism**.

## 2. Design Goals & Non-Goals

| Goals | Non-Goals (v1.0) |
|---|---|
| Model-agnostic OCR layer; swap SOTA models without touching the pipeline | Training / fine-tuning models (see roadmap) |
| Correct handling of 横書き and 縦書き reading order | Historical cursive script (くずし字) |
| LLM correction that is **provably bounded** (guard + diff) | Fully automatic decisions without human review for critical fields |
| On-premise capable (APPI-friendly) | Real-time (<100 ms) latency |
| Reproducible evaluation (CER) | Handwritten math / chemical formulas |

## 3. System Overview

```text
                 ┌──────────────┐      ┌───────────────────┐      ┌───────────────────────┐
  bytes ───────► │   utils      │ ───► │  preprocessor     │ ───► │  ocr_engine           │
 (img/pdf/tiff)  │ load_document│ BGR  │ ImagePreprocessor │ Pre- │  OCREngine            │
                 │   _pages()   │ page │   .process()      │ proc │   └─ BaseOCRBackend   │
                 └──────────────┘      └───────────────────┘ Result└──────────┬────────────┘
                                                                              │ OCRResult
                                                                              ▼
                 ┌──────────────┐      ┌───────────────────────────────────────────────────┐
  JSON / TXT ◄── │   app.py     │ ◄─── │  postprocessor                                    │
  Streamlit UI   │ build_export │      │  JapanesePostCorrector ── BaseLLMClient           │
                 └──────────────┘      │   chunk → prompt → LLM → parse → guard → merge    │
                                       └───────────────────────────────────────────────────┘
```

**Module responsibilities (single responsibility principle):**

| Module | Depends on | Heavy deps | Responsibility |
|---|---|---|---|
| `config.py` | dotenv | — | Typed, validated, immutable settings (frozen dataclasses) |
| `src/utils.py` | OpenCV, Pillow, PyMuPDF | — | I/O, format conversion, logging, timing, CER |
| `src/preprocessor.py` | OpenCV, NumPy | — | Image enhancement, deskew, direction, layout, reading order |
| `src/ocr_engine.py` | torch, transformers, openai | **lazy** | VLM inference behind a common interface |
| `src/postprocessor.py` | openai, anthropic | **lazy** | Guarded LLM correction, diffing |
| `app.py` | Streamlit | — | Orchestration, caching, visualization, export |

Heavy libraries are imported **inside** backend constructors, so the lightweight modules (and their tests) run in a CPU-only CI container without PyTorch.

## 4. Stage 1 — Pre-processing & Layout Analysis

```text
 BGR ─► resize(max_side) ─► gray ─► NLM denoise ─► CLAHE ─► adaptive threshold (INV)
                                                               │
            ┌──────────────────────────────────────────────────┘
            ▼
   estimate_skew(): projection-profile search  ──► rotate(gray, angle, expand=True)
            │
            ▼
   content_bounding_box() ──► crop ──► detect_orientation() ──► analyze_layout()
                                                                   │
                                                                   ▼
                                           merge overlapping boxes ─► reading order sort
```

### 4.1 Image enhancement

| Step | Technique | Rationale |
|---|---|---|
| Resize | `INTER_AREA`, longest side ≤ 2048 px | VLMs downsample internally anyway; bounding input size bounds latency and VRAM. Never upscale. |
| Denoise | `fastNlMeansDenoising` (h=10) | Preserves thin strokes (はね・はらい) better than Gaussian/median blur. |
| Contrast | CLAHE (clip 2.0, 8×8 tiles) | Local equalization neutralizes phone-camera shadows and yellowed paper. |
| Binarize | Adaptive Gaussian threshold, window ∝ image size | Global Otsu fails under uneven lighting. Output is used **for analysis only** by default. |

> **Why not feed the binary image to the VLM?** Modern VLMs are pre-trained on natural images; binarization destroys stroke-width and pressure information that helps distinguish similar Kanji. The option `PREPROCESS_BINARIZE_FOR_OCR=true` exists for very dirty or patterned backgrounds.

### 4.2 Deskew — projection profile search

For a candidate angle θ, rotate the binary map and compute the row-sum and column-sum profiles. When lines (横書き) or columns (縦書き) are aligned, one of these profiles becomes maximally "peaky":

```text
score(θ) = max( Var(Σ_x I_θ(x, y)),  Var(Σ_y I_θ(x, y)) )
θ* = argmax score(θ)   — coarse grid 0.5° over [−10°, +10°], then 0.1° refinement
```

Taking the `max` of both axes makes the method **direction-agnostic**, which `minAreaRect`-based approaches are not (they are also unstable on sparse handwriting and suffer from OpenCV's angle-convention changes across versions). Corrections < 0.1° are skipped to avoid interpolation blur.

### 4.3 Writing-direction detection (横書き vs 縦書き)

1. Estimate character size *s* as the median bounding-box size of connected components.
2. Dilate with a horizontal line kernel `(1.2s × 1)` and, separately, a vertical kernel `(1 × 1.2s)`.
3. The direction whose dilation merges characters into **fewer** components is the writing direction (`vertical` if `N_v < 0.85 · N_h`).

This is O(pixels), needs no model, and can be overridden with `PREPROCESS_TEXT_DIRECTION`.

### 4.4 Layout analysis & Japanese reading order

Anisotropic dilation joins characters into lines/columns and adjacent lines into paragraph blocks (`2s` along the writing direction, `0.8s` across). External contours → bounding boxes → iterative overlap merge → sort:

```text
 横書き (horizontal)                    縦書き (vertical)
 ┌─────────────────────────┐            ┌─────────────────────────┐
 │ [1] ───────────►        │            │  [3]   [2]   [1]        │
 │ [2] ───────────►        │            │   │     │     │         │
 │ [3] ──────► [4] ──────► │            │   ▼     ▼     ▼         │
 └─────────────────────────┘            └─────────────────────────┘
 rows top→bottom, then left→right       columns right→left, then top→bottom
```

Rows/columns are grouped with a tolerance equal to the estimated character size, which is robust to slightly wavy handwritten baselines.

## 5. Stage 2 — Vision-Language OCR Engine

### 5.1 Why VLMs instead of a classic detector + CRNN?

| Aspect | Classic (detector + CRNN/CTC) | Vision-Language Model |
|---|---|---|
| Character set | Fixed output head; rare Kanji often missing | Tokenizer covers full Unicode |
| Language context | None (or separate LM) | Built-in; resolves many confusables at decode time |
| Layout | Needs separate detection/ordering | Understands page structure end-to-end |
| Direction | Needs rotation-specific models for 縦書き | Handles both with prompting |
| Cost | Very cheap | Expensive (GPU, latency) |

VLMs dominate on accuracy for messy, mixed-script handwriting; the pipeline mitigates their cost via input-size control, greedy decoding, and optional remote serving.

### 5.2 Backend abstraction (Strategy + Factory)

```text
                    ┌───────────────────────────┐
                    │ BaseOCRBackend (ABC)      │
                    │  + recognize(img, prompt) │
                    └─────────────┬─────────────┘
          ┌───────────────────────┼─────────────────────────┬──────────────────────────┐
          ▼                       ▼                         ▼                          ▼
 HFVisionLanguageBackend    GOTOCR2Backend      OpenAICompatibleVisionBackend   (your backend)
   ├─ QwenVLBackend         (processor-native    (REST, base64 PNG,
   └─ GLMOCRBackend          prompt-free API)     temperature 0)
```

`create_ocr_backend(config)` looks up a registry dict; `OCREngine` accepts an injected backend for unit testing (dependency injection).

### 5.3 Supported models (qualitative comparison)

| Model | Approx. params | Strengths | Caveats |
|---|---|---|---|
| Qwen2.5-VL-3B / 7B / 72B-Instruct | 3B / 7B / 72B | Strong multilingual (incl. Japanese) reading, instruction following, both writing directions | 7B needs ~16–24 GB VRAM in bf16; possible repetition loops on long pages (mitigated by `repetition_penalty`) |
| GLM-OCR (Z.ai) | ~0.9B (see model card) | OCR-specialized, lightweight, fast | Uses short task prompts (`Text Recognition:`); requires a recent `transformers` |
| GOT-OCR 2.0 (`-hf`) | ~0.6B | Very fast end-to-end OCR, small footprint | Training data mostly English/Chinese; Japanese handwriting accuracy lower → rely on Stage 3 |

> Parameter counts are approximate; consult each model card for exact figures, licenses, and supported `transformers` versions.

### 5.4 Inference details

- **Prompting:** the default prompt is written *in Japanese* and forbids summarization, translation, and "fixing" the writer's own typos; unreadable characters are emitted as `〓` (the traditional Japanese typesetting placeholder, ゲタ記号), which Stage 3 may resolve only when context is unambiguous.
- **Decoding:** greedy (`do_sample=False`) for determinism, `repetition_penalty=1.05` to break degenerate loops.
- **Precision:** bf16 on Ampere+ GPUs, fp16 on older GPUs and Apple MPS, fp32 on CPU; `device_map="auto"` for multi-GPU sharding.
- **Qwen visual tokens:** `max_pixels = 1280·28·28` caps visual tokens (~1280), balancing stroke detail vs. latency.
- **Modes:** `page` (default; single call, the VLM handles layout) vs `block` (one call per Stage-1 block in reading order; useful for very dense pages or low-resolution models).
- **Sanitization:** strip Markdown code fences, normalize line endings, collapse redundant blank lines.

## 6. Stage 3 — LLM Contextual Post-Correction

### 6.1 Why a separate correction stage?

Even the best VLM confuses glyph pairs whose disambiguation needs **semantic** context: 「未払い」 vs 「末払い」, 「土曜日」 vs 「士曜日」, 「シート」 vs 「ツート」. A text-only LLM with a strong Japanese prior fixes these cheaply — but LLMs also love to *rewrite*, which is unacceptable for records.

### 6.2 Three layers of defense against hallucination

```text
         OCR text
            │
            ▼
   ┌─────────────────┐   ≤ LLM_MAX_CHARS_PER_CHUNK, split on line boundaries
   │ 1. Chunking     │   → bounded context, localized failures
   └────────┬────────┘
            ▼
   ┌─────────────────┐   Japanese system prompt: fix only OCR errors; never paraphrase,
   │ 2. Strict prompt│   never touch numbers/dates/names unless obviously misread;
   │    + JSON schema│   keep line breaks; output {"corrected_text", "corrections[]"}
   └────────┬────────┘
            ▼
   ┌─────────────────┐   edit_ratio = 1 − SequenceMatcher(orig, corr).ratio()
   │ 3. Edit-ratio   │   edit_ratio > threshold  ─►  REJECT chunk, keep original
   │    guard        │   edit_ratio ≤ threshold  ─►  ACCEPT
   └────────┬────────┘
            ▼
   Verified diff (difflib opcodes) + LLM self-reported rationale  ─► human review
```

- **Fail-safe:** any network error, rate limit, or malformed JSON returns the original chunk and records the error. The pipeline never fails because of the LLM.
- **Two change logs:** `llm_corrections` (self-reported, may be incomplete) and `verified_changes` (computed by diff — the ground truth of what actually changed). Reviewers should trust the latter.
- **Domain context:** an optional hint such as 「医療問診票」 or 「請求書」 steers vocabulary choice (e.g., medical terms).
- **Temperature 0** for reproducibility.

### 6.3 Provider abstraction

`BaseLLMClient.complete(system, user) -> str` with `OpenAIChatClient` (OpenAI and any compatible server: vLLM, Ollama, Azure) and `AnthropicChatClient`. Setting `OPENAI_BASE_URL` to a local server makes Stage 3 fully on-premise.

## 7. Benchmark Methodology

> ⚠️ **This repository deliberately does not publish accuracy numbers it has not measured.** Handwriting OCR accuracy varies enormously with document type, writer, capture device, and domain vocabulary. Use the protocol below on your own labeled data.

### 7.1 Metric: Character Error Rate (CER)

```text
CER = (S + D + I) / N
  S = substitutions, D = deletions, I = insertions (character-level Levenshtein)
  N = number of characters in the reference
```

Both strings are normalized before scoring (`src/utils.normalize_for_eval`): Unicode **NFKC** (unifies full-/half-width forms) and removal of whitespace/newlines (Japanese has no word spacing, and line-break placement is not a recognition error). Accuracy is reported as `1 − CER`.

### 7.2 Recommended evaluation protocol

1. Collect ≥ 100 pages representative of production (form types, writers, scanners vs phones).
2. Transcribe ground truth with double-annotation and adjudication.
3. Report CER for each configuration **before and after** Stage 3, plus the guard rejection rate.
4. Break down errors by script class (Kanji / Hiragana / Katakana / digits / Latin / symbols) and by field type (names, addresses, dates, amounts).
5. Report latency percentiles (p50/p95) per page and hardware used.

Public resources useful for character-level sanity checks include the **ETL Character Database** (AIST; isolated handwritten characters) — note that isolated-character sets do not reflect page-level difficulty.

### 7.3 Results template

| Configuration | Hardware | CER raw ↓ | CER + LLM ↓ | Guard rejections | p50 latency / page |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B · page | *your GPU* | — | — | — | — |
| Qwen2.5-VL-3B · page | *your GPU* | — | — | — | — |
| GLM-OCR · page | *your GPU* | — | — | — | — |
| GOT-OCR 2.0 · page | *your GPU* | — | — | — | — |
| Qwen2.5-VL-7B · block | *your GPU* | — | — | — | — |

The Streamlit **Evaluation** tab computes CER interactively for a single page when you paste ground truth.

## 8. Performance & Scaling

| Lever | Effect |
|---|---|
| `PREPROCESS_MAX_SIDE` / `OCR_QWEN_MAX_PIXELS` | Fewer visual tokens → lower latency & VRAM, at some cost to small-stroke detail |
| Smaller model (3B vs 7B) | ~2× faster, lower accuracy on hard handwriting |
| `OCR_MODE=page` | 1 VLM call per page instead of N blocks |
| `openai-compatible` + vLLM | Continuous batching, paged attention, horizontal scaling of GPU workers |
| `st.cache_resource` | Model loaded once per process, reused across requests |
| Stage 3 chunking | Parallelizable; small chunks keep LLM latency and cost predictable |

**Target production topology:**

```text
  Browser ─► Streamlit / FastAPI (CPU pods, stateless)
                   │  OpenAI-compatible HTTP
                   ▼
             vLLM VLM servers (GPU pool, autoscaled)   ─┐
             LLM for post-correction (API or local)    ─┴─► results store (DB / object storage)
```

## 9. Security, Privacy & Compliance

- Handwritten forms usually contain **personal information** (氏名, 住所, 生年月日, medical data). Under Japan's APPI (個人情報保護法), sending such data to external APIs may require consent or contractual safeguards.
- The system can run **entirely on-premise**: local VLM (transformers or vLLM) + local LLM via `OPENAI_BASE_URL` (Ollama/vLLM), or `LLM_PROVIDER=none`.
- Secrets are read from environment variables only; `.env` is git-ignored; the container runs as a non-root user.
- Uploaded documents are processed in memory; nothing is persisted unless the user downloads the export.

## 10. Failure Modes & Mitigations

| Failure mode | Symptom | Mitigation |
|---|---|---|
| VLM repetition loop | Same phrase repeated until `max_new_tokens` | `repetition_penalty`, token cap, block mode |
| Wrong reading order in 縦書き | Columns read left→right | Direction detection; override `PREPROCESS_TEXT_DIRECTION=vertical` |
| LLM paraphrasing | Fluent but different sentence | Strict prompt + edit-ratio guard + verified diff |
| LLM API outage | Timeouts / 429 | Fail-safe returns raw OCR, errors surfaced in UI/JSON |
| Over-aggressive deskew | Rotated tables/stamps | Bounded search range, 0.1° dead zone, toggle in UI |
| Stamps (印鑑) over text | Red overlaps confuse OCR | Future work: color-based stamp removal before grayscale |
| OOM on large PDFs | Crash on many pages | `MAX_PDF_PAGES`, `PDF_DPI`, image size caps |

## 11. Extensibility Guide

**Add a new OCR model:**

```python
# src/ocr_engine.py
class MyNewBackend(HFVisionLanguageBackend):
    name = "my-model"

_BACKEND_REGISTRY["my-model"] = MyNewBackend
```

Then add `"my-model"` to `SUPPORTED_OCR_BACKENDS` and `DEFAULT_MODEL_IDS` in `config.py`. No change to the UI or pipeline is needed.

**Add a new LLM provider:** implement `BaseLLMClient.complete()` and register it in `create_llm_client()`.

## 12. Architecture Decision Records

| ADR | Decision | Alternatives considered | Reason |
|---|---|---|---|
| 001 | VLM-based OCR as the core recognizer | Tesseract `jpn`, detector + CRNN | Far better on handwriting and mixed scripts; context-aware |
| 002 | Keep classic CV pre-processing | End-to-end VLM only | Cheap, deterministic gains on phone photos; provides layout for block mode and UI |
| 003 | LLM correction as a *separate*, guarded stage | Ask the VLM to "correct while reading" | Separation of concerns, auditability, independent model choice |
| 004 | Edit-ratio guard instead of confidence scores | Token log-prob thresholds | Provider-agnostic (works with any API), simple to reason about |
| 005 | Frozen dataclasses + env vars for config | YAML, pydantic-settings | Zero extra dependencies, immutable, 12-factor |
| 006 | Lazy imports for torch/transformers | Top-level imports | Fast startup, CPU-only CI, API-only deployments without PyTorch overhead at import time |

---

## 🇻🇳 Tóm tắt tiếng Việt

**Mục tiêu:** chuyển tài liệu viết tay tiếng Nhật thành văn bản số **trung thực** (không "bịa" nội dung), đúng thứ tự đọc, kèm metadata để kiểm tra lại.

**Tầng 1 – Tiền xử lý (OpenCV):** khử nhiễu NLM giữ nét chữ mảnh, CLAHE xử lý bóng đổ, chỉnh nghiêng bằng *projection profile* (thử các góc, chọn góc làm biểu đồ tổng pixel theo hàng/cột dao động mạnh nhất — hoạt động cho cả chữ ngang lẫn dọc). Nhận diện 横書き/縦書き bằng giãn nở hình thái học theo hai phương, sau đó phân khối và sắp xếp thứ tự đọc chuẩn Nhật (chữ dọc: cột phải sang trái, trên xuống dưới).

**Tầng 2 – OCR bằng VLM:** kiến trúc Strategy + Factory cho phép thay Qwen2.5-VL, GLM-OCR, GOT-OCR 2.0 hoặc model qua API vLLM chỉ bằng cấu hình. Prompt tiếng Nhật, giải mã greedy, phạt lặp, ký tự không đọc được ghi là 〓.

**Tầng 3 – LLM hậu xử lý:** chia đoạn, prompt nghiêm ngặt chỉ sửa lỗi nhận dạng (未/末, シ/ツ...), đầu ra JSON có lý do. **Guard edit-ratio** từ chối bản sửa nếu thay đổi quá ngưỡng; lỗi API thì giữ nguyên bản OCR (fail-safe). Diff được tính độc lập để con người kiểm duyệt.

**Đánh giá:** dùng CER sau khi chuẩn hóa NFKC. Dự án cố ý **không công bố số liệu chưa đo** — hãy chạy theo quy trình ở mục 7 trên dữ liệu thật của bạn.

**Bảo mật:** có thể chạy hoàn toàn on-premise, phù hợp luật APPI của Nhật Bản.

# 📜 Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `requirements-gpu.txt`: PyTorch/Transformers split out; `requirements.txt` is now a lightweight core set (API-based OCR) suitable for Streamlit Community Cloud.
- Streamlit Secrets support: root-level secrets are exported to environment variables before configuration is loaded.
- `.streamlit/secrets.toml.example` and a no-GPU Gemini preset in `.env.example`.
- `.dockerignore` to keep secrets, virtualenvs and weights out of Docker images.
- `INSTALL_MODE=lite|full` option in `run.sh`.

### Changed
- Dockerfile now installs `requirements-gpu.txt` (local VLM inference).

### Planned
- Learned layout detection (tables, form fields, stamps / 印鑑).
- Key-value extraction into typed schemas (氏名, 住所, 生年月日, 金額).
- FastAPI REST service with async batch processing.
- Evaluation CLI with per-script-class error breakdown.

## [1.0.0] - 2026-09-27

🎉 **First public release** — a production-oriented 3-stage pipeline for Japanese handwritten document OCR.

### Added

#### Stage 1 — Pre-processing & Layout Analysis (`src/preprocessor.py`)
- Non-Local Means denoising and CLAHE contrast enhancement for phone photos and aged paper.
- Adaptive binarization with image-size-aware window.
- Direction-agnostic **projection-profile deskew** (coarse 0.5° + fine 0.1° search).
- Automatic border cropping to the content bounding box.
- **横書き / 縦書き detection** via anisotropic morphological dilation.
- Morphological layout analysis with overlap merging and **Japanese reading order** (right-to-left columns for vertical text).
- Block overlay visualization with reading-order numbers.

#### Stage 2 — Vision-Language OCR Engine (`src/ocr_engine.py`)
- Pluggable backend architecture (Strategy + Factory + registry).
- Backends: **Qwen2.5-VL**, **GLM-OCR**, **GOT-OCR 2.0 (`-hf`)**, and **OpenAI-compatible API** (vLLM / SGLang / LMDeploy).
- Automatic device (CUDA › MPS › CPU) and dtype (bf16 / fp16 / fp32) selection; multi-GPU via `device_map="auto"`.
- `page` and `block` OCR modes.
- Japanese OCR prompt with `〓` placeholder for illegible characters; greedy decoding and repetition penalty.
- Output sanitization (code-fence stripping, newline normalization).

#### Stage 3 — LLM Contextual Post-Correction (`src/postprocessor.py`)
- OpenAI (and compatible servers) and Anthropic providers.
- Strict Japanese system prompt targeting confusable glyphs (未/末, シ/ツ, カ/力, …) with structured JSON output and per-edit rationale.
- Line-aware chunking for long documents.
- **Hallucination guard** based on character-level edit ratio.
- Fail-safe behavior on API/JSON errors (original text preserved).
- Diff-verified change log and HTML diff rendering.

#### Application & Tooling
- Streamlit web UI: step-by-step pre-processing gallery, layout overlay, raw vs corrected text, character diff, CER evaluation tab, JSON/TXT export.
- Multi-format input: PNG, JPG, WebP, BMP, multi-page TIFF, PDF (PyMuPDF).
- Typed, validated 12-factor configuration (`config.py`, `.env.example`).
- Utilities: EXIF-aware image loading, CER metric with NFKC normalization, timers, structured logging.
- `run.sh` for install / start / docker workflows.
- Dockerfile with CPU/GPU build argument, non-root user, and healthcheck.
- Documentation: README, ARCHITECTURE (design deep dive, ADRs, benchmark methodology), CONTRIBUTING, MIT LICENSE.

### Security
- Secrets loaded only from environment variables; `.env` and model weights excluded via `.gitignore`.
- Fully on-premise operation supported (local VLM + local LLM) for APPI-sensitive deployments.

[Unreleased]: https://github.com/<your-username>/japanese-handwritten-ocr/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/<your-username>/japanese-handwritten-ocr/releases/tag/v1.0.0

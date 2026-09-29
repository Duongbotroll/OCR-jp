# 🤝 Contributing Guide

Thank you for your interest in improving **Japanese Handwritten Document OCR**! This guide explains how to set up a development environment, the coding standards we follow, and how to submit changes.

> 🇻🇳 Tóm tắt tiếng Việt ở cuối tài liệu.

## Code of Conduct

Be respectful, constructive, and inclusive. Harassment of any kind is not tolerated. Assume good intent and focus feedback on the code, not the person.

## Ways to Contribute

- 🐛 **Report bugs** — open an issue with steps to reproduce, expected vs. actual behavior, environment (OS, Python, GPU, `transformers` version) and, if possible, a **non-sensitive** sample image.
- 💡 **Propose features** — open an issue describing the use case before writing large amounts of code.
- 🤖 **Add an OCR backend or LLM provider** — see [ARCHITECTURE.md § Extensibility](ARCHITECTURE.md#11-extensibility-guide).
- 📊 **Share benchmark results** — CER measurements on public or synthetic data following [ARCHITECTURE.md § 7](ARCHITECTURE.md#7-benchmark-methodology).
- 📝 **Improve documentation** — typos, clarifications, translations.

> ⚠️ **Never attach real personal documents** (forms containing names, addresses, medical data) to issues or PRs. Use synthetic samples.

## Development Setup

```bash
git clone https://github.com/<your-username>/japanese-handwritten-ocr.git
cd japanese-handwritten-ocr
./run.sh install            # creates .venv and installs requirements
source .venv/bin/activate
pip install ruff mypy pytest
```

## Branching & Commit Convention

- Branch from `main` using: `feat/<short-name>`, `fix/<short-name>`, `docs/<short-name>`, `refactor/<short-name>`.
- Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
feat(ocr): add PaddleOCR-VL backend
fix(preprocessor): correct reading order for single-column vertical text
docs(readme): add GPU memory table
refactor(postprocessor): extract diff rendering helper
test(utils): add CER edge cases
```

## Coding Standards

| Rule | Details |
|---|---|
| Style | **PEP 8**, max line length **100**, enforced with `ruff` |
| Typing | Type hints on every public function/method; `from __future__ import annotations` at the top of modules |
| Docstrings | Every module, class, and public function has a docstring (Google style: `Args`, `Returns`, `Raises`) |
| Comments | Explanatory comments are written in **Vietnamese** (project convention); identifiers and docs in English |
| Imports | Standard library → third-party → local; **heavy libraries (torch, transformers, SDKs) are imported lazily** inside the classes that need them |
| Configuration | No hard-coded paths, model IDs, or secrets — add new settings to `config.py` and `.env.example` |
| Errors | Raise specific exceptions with actionable messages; Stage 3 must stay fail-safe (never crash the pipeline) |
| Logging | Use `setup_logger(__name__)`; never `print` in library code; never log API keys or document text at INFO level |
| Determinism | OCR/LLM inference defaults must stay deterministic (greedy decoding, temperature 0) |

### Quality checks (run before every PR)

```bash
ruff check . --line-length 100
ruff format --check . --line-length 100
mypy src config.py --ignore-missing-imports
python -m compileall -q src config.py app.py
pytest -q                     # when tests are present
```

### Writing tests

- Put tests in `tests/`, mirroring `src/` (e.g., `tests/test_preprocessor.py`).
- Tests **must not download models or call external APIs**. Inject fakes instead:

```python
from src.ocr_engine import BaseOCRBackend, OCREngine

class FakeBackend(BaseOCRBackend):
    name = "fake"
    def recognize(self, image, prompt=""):
        return "テスト"

engine = OCREngine(config, backend=FakeBackend(config))
```

- The same pattern applies to `JapanesePostCorrector(config, client=FakeLLMClient())`.

## Pull Request Process

1. Make sure all quality checks pass and the Streamlit app starts (`./run.sh start`).
2. Update `README.md` / `ARCHITECTURE.md` if behavior or configuration changes.
3. Add an entry under **[Unreleased]** in `CHANGELOG.md`.
4. Open a PR with: summary, motivation, screenshots of UI changes, and benchmark impact if relevant.
5. At least one maintainer approval is required; squash-merge is used to keep history clean.

### PR checklist

- [ ] Follows PEP 8, type hints and docstrings present
- [ ] No secrets, weights, or personal data committed
- [ ] New config values documented in `.env.example`
- [ ] CHANGELOG updated
- [ ] Tested on CPU at least; GPU-specific behavior noted

## Release Process (maintainers)

1. Move **[Unreleased]** entries into a new version section in `CHANGELOG.md` following [Semantic Versioning](https://semver.org/).
2. Bump `__version__` in `src/__init__.py`.
3. Tag: `git tag -a vX.Y.Z -m "vX.Y.Z" && git push --tags`.

---

## 🇻🇳 Tóm tắt tiếng Việt

- **Đóng góp:** báo lỗi, đề xuất tính năng, thêm backend OCR/LLM, chia sẻ kết quả benchmark, cải thiện tài liệu. **Tuyệt đối không** đính kèm tài liệu thật chứa thông tin cá nhân.
- **Quy ước:** nhánh `feat/…`, `fix/…`; commit theo Conventional Commits.
- **Chuẩn code:** PEP 8 (tối đa 100 ký tự/dòng), type hints, docstring kiểu Google, comment giải thích bằng tiếng Việt, import trễ thư viện nặng, không hard-code cấu hình.
- **Kiểm thử:** không tải model hay gọi API thật trong test — dùng backend/client giả lập (dependency injection).
- **PR:** chạy `ruff`, `mypy`, `compileall`, cập nhật tài liệu và CHANGELOG, cần ít nhất một maintainer duyệt.

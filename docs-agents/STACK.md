# Stack

One line per choice; parentheses only if extra detail is needed.

- **Language:** Python (≥3.10, tested through 3.13)
- **Packaging:** setuptools + wheel; published to PyPI as `datafast`
- **Dependency manager:** uv (`uv.lock`)
- **LLM calling:** LiteLLM (unified gateway); provider SDKs `openai`, `anthropic`, `google-generativeai`; `botocore` for AWS
- **Structured outputs:** Instructor + Pydantic
- **Data:** Hugging Face `datasets`
- **UI:** Gradio
- **Logging:** Loguru
- **LLM observability:** Langfuse (optional extra, via LiteLLM)
- **Config:** python-dotenv (`.env`)
- **Tests:** pytest (run via `.venv/bin/pytest`)
- **Lint/format:** Ruff (line length 88)
- **Docs:** MkDocs + Material theme

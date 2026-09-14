"""Test-only environment defaults independent of a developer's local .env."""

from __future__ import annotations

import os

os.environ["LLM_BASE_URL"] = "http://localhost:11434/v1"
os.environ["LLM_API_KEY"] = "ollama"
os.environ["LLM_MODEL"] = "smollm2:1.7b"

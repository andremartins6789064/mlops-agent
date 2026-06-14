from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.domain.entities import Notebook, Pipeline


class INotebookParser(ABC):
    @abstractmethod
    def parse(self, notebook_path: str) -> Notebook:
        """Parse an .ipynb file and return the domain Notebook representation."""


class ILLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Generate a completion for the provided prompt."""


class IExporter(ABC):
    @abstractmethod
    def export(self, pipeline: Pipeline, output_dir: str, **kwargs: Any) -> str:
        """Export pipeline artifacts and return the generated file path."""

"""Utilities for extracting structured content from LLM responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Value extracted from an LLM response and the extraction method used."""

    value: Any | None
    method: str


def parse_json_object(raw_response: str) -> ParseResult:
    """Extract a JSON object from a direct, fenced, or noisy response."""
    response = _remove_think_tags(raw_response).strip()
    if not response:
        return ParseResult(value=None, method="failed")

    direct_value = _load_json_object(response)
    if direct_value is not None:
        return ParseResult(value=direct_value, method="direct")

    for fenced_content in _fenced_blocks(response):
        fenced_value = _first_json_object(fenced_content)
        if fenced_value is not None:
            return ParseResult(value=fenced_value, method="fenced")

    scanned_value = _first_json_object(response)
    if scanned_value is not None:
        return ParseResult(value=scanned_value, method="scanned")

    return ParseResult(value=None, method="failed")


def parse_python_block(raw_response: str) -> ParseResult:
    """Extract Python source from a fenced block or a raw code response."""
    response = _remove_think_tags(raw_response).strip()
    if not response:
        return ParseResult(value=None, method="failed")

    for language, content in _fenced_blocks_with_language(response):
        if language in {"", "python", "py"} and _looks_like_python(content):
            return ParseResult(value=content.strip(), method="fenced")

    if _looks_like_python(response):
        return ParseResult(value=response, method="raw")

    return ParseResult(value=None, method="failed")


def _load_json_object(candidate: str) -> dict[str, Any] | None:
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _first_json_object(text: str) -> dict[str, Any] | None:
    for start, end in _balanced_object_ranges(text):
        value = _load_json_object(text[start:end])
        if value is not None:
            return value
    return None


def _balanced_object_ranges(text: str) -> list[tuple[int, int]]:
    """Return balanced object ranges while ignoring braces inside strings."""
    ranges: list[tuple[int, int]] = []
    stack: list[int] = []
    in_string = False
    escaped = False

    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            stack.append(index)
        elif character == "}" and stack:
            start = stack.pop()
            if not stack:
                ranges.append((start, index + 1))

    return ranges


def _fenced_blocks(text: str) -> list[str]:
    return [content for _, content in _fenced_blocks_with_language(text)]


def _fenced_blocks_with_language(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"```([^\n`]*)\n?(.*?)```", re.DOTALL)
    return [
        (match.group(1).strip().lower(), match.group(2))
        for match in pattern.finditer(text)
    ]


def _looks_like_python(source: str) -> bool:
    return bool(re.search(r"\b(?:def|class|import|from)\s+", source))


def _remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

from src.shared.llm_parsing import parse_json_object, parse_python_block


def test_parse_json_object_accepts_direct_json() -> None:
    result = parse_json_object('{"stage": "training", "cells": [1, 2]}')

    assert result.value == {"stage": "training", "cells": [1, 2]}
    assert result.method == "direct"


def test_parse_json_object_handles_nested_json_in_markdown() -> None:
    response = (
        "Here is the result:\n"
        '```json\n{"modules": {"training": {"functions": ["fit"]}}}\n```'
    )

    result = parse_json_object(response)

    assert result.value == {"modules": {"training": {"functions": ["fit"]}}}
    assert result.method == "fenced"


def test_parse_json_object_scans_text_before_and_after_object() -> None:
    result = parse_json_object(
        'I thought about it first. {"answer": {"value": 42}} Done.'
    )

    assert result.value == {"answer": {"value": 42}}
    assert result.method == "scanned"


def test_parse_json_object_rejects_truncated_nested_json() -> None:
    result = parse_json_object('{"answer": {"value": 42}')

    assert result.value is None
    assert result.method == "failed"


def test_parse_json_object_ignores_think_tags() -> None:
    result = parse_json_object(
        '<think>{"intermediate": true}</think>\n{"answer": "ok"}'
    )

    assert result.value == {"answer": "ok"}
    assert result.method == "direct"


def test_parse_json_object_rejects_empty_response() -> None:
    result = parse_json_object("")

    assert result.value is None
    assert result.method == "failed"


def test_parse_python_block_extracts_fenced_code() -> None:
    result = parse_python_block(
        "Here is the module:\n```python\ndef train_model() -> int:\n    return 1\n```\n"
    )

    assert result.value == "def train_model() -> int:\n    return 1"
    assert result.method == "fenced"


def test_parse_python_block_accepts_raw_code() -> None:
    result = parse_python_block("def train_model() -> int:\n    return 1")

    assert result.value == "def train_model() -> int:\n    return 1"
    assert result.method == "raw"


def test_parse_python_block_rejects_empty_or_non_python_response() -> None:
    assert parse_python_block("").method == "failed"
    assert parse_python_block("I cannot help with that.").value is None

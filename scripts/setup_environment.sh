#!/usr/bin/env bash

set -e

echo "Installing project dependencies..."
uv sync

echo "Installing pre-commit hooks..."
uv run pre-commit install

echo "Environment ready."

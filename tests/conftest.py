"""Shared pytest fixtures/config.

Skips the whole test session with a clear message if the (large,
gitignored) ONNX model file isn't present on disk -- keeps the test suite
runnable in environments that haven't fetched the model.
"""
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "model", "zipformer_p_arabic_v3.int8.onnx")
TOKENS_PATH = os.path.join(REPO_ROOT, "model", "tokens.txt")
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def pytest_collection_modifyitems(config, items):
    if not os.path.exists(MODEL_PATH):
        skip_marker = pytest.mark.skip(
            reason=f"model file not found at {MODEL_PATH!r} -- "
                   "copy zipformer_p_arabic_v3.int8.onnx into model/ before running tests"
        )
        for item in items:
            item.add_marker(skip_marker)

import os

import pytest


@pytest.fixture
def sample_corpus_path() -> str:
    """Path to the synthetic sample raw corpus jsonl file."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "sample_corpus.jsonl")


@pytest.fixture
def tmp_processed_dir(tmp_path) -> str:
    """Temporary directory for processed files."""
    d = tmp_path / "processed"
    d.mkdir()
    return str(d)

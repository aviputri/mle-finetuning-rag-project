"""
Test configuration and fixtures for pytest.
"""

import pytest

from src import config
from src.data import load_documents, load_evaluation_cases, load_training_examples
from src.retrieval import Retriever


@pytest.fixture(scope="session")
def documents():
    return load_documents(config.KB_DIR)


@pytest.fixture(scope="session")
def examples():
    return load_training_examples(config.TRAIN_FILE)


@pytest.fixture(scope="session")
def cases():
    return load_evaluation_cases(config.EVAL_FILE)


@pytest.fixture(scope="session")
def retriever():
    # Built once for the whole test run (embedding model is cached locally)
    return Retriever()
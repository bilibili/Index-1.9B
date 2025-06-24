"""Shared pytest fixtures and configuration for all tests."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.
    
    Yields:
        Path: Path to the temporary directory.
    """
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for testing.
    
    Returns:
        Dict[str, Any]: Mock configuration with common settings.
    """
    return {
        "model_name": "test_model",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 10,
        "device": "cpu",
        "seed": 42,
        "output_dir": "/tmp/test_output",
        "data_dir": "/tmp/test_data",
        "max_length": 512,
        "num_workers": 2,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 100,
        "logging_steps": 50,
        "save_steps": 1000,
        "eval_steps": 500,
    }


@pytest.fixture
def mock_model():
    """Create a mock model for testing.
    
    Returns:
        MagicMock: Mock model object with common methods.
    """
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.train = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.parameters = MagicMock(return_value=[])
    model.state_dict = MagicMock(return_value={})
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing.
    
    Returns:
        MagicMock: Mock tokenizer object with common methods.
    """
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=[101, 102, 103])
    tokenizer.decode = MagicMock(return_value="test text")
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.bos_token_id = 2
    tokenizer.__call__ = MagicMock(return_value={
        "input_ids": [[101, 102, 103]],
        "attention_mask": [[1, 1, 1]],
    })
    return tokenizer


@pytest.fixture
def sample_text_data() -> list[str]:
    """Provide sample text data for testing.
    
    Returns:
        list[str]: List of sample text strings.
    """
    return [
        "This is a test sentence.",
        "Another example for testing purposes.",
        "Machine learning models need good test data.",
        "Python testing with pytest is efficient.",
        "Natural language processing tasks require validation.",
    ]


@pytest.fixture
def sample_dataset(sample_text_data: list[str]) -> Dict[str, list]:
    """Create a sample dataset dictionary.
    
    Args:
        sample_text_data: List of sample text strings.
        
    Returns:
        Dict[str, list]: Dataset dictionary with train/val/test splits.
    """
    return {
        "train": sample_text_data[:3],
        "validation": sample_text_data[3:4],
        "test": sample_text_data[4:],
    }


@pytest.fixture
def mock_api_response() -> Dict[str, Any]:
    """Mock API response for testing API interactions.
    
    Returns:
        Dict[str, Any]: Mock API response data.
    """
    return {
        "id": "test-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    }


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test.
    
    This fixture automatically runs before each test to ensure
    a clean environment state.
    """
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture log messages during tests.
    
    Args:
        caplog: pytest's built-in log capture fixture.
        
    Returns:
        caplog: The log capture fixture for assertions.
    """
    caplog.set_level("DEBUG")
    return caplog


@pytest.fixture
def mock_file_content() -> str:
    """Provide mock file content for testing file operations.
    
    Returns:
        str: Mock file content.
    """
    return """# Test Configuration File
model_name: test_model
batch_size: 32
learning_rate: 0.0001
"""


@pytest.fixture
def mock_json_data() -> Dict[str, Any]:
    """Provide mock JSON data for testing.
    
    Returns:
        Dict[str, Any]: Mock JSON data structure.
    """
    return {
        "version": "1.0.0",
        "data": [
            {"id": 1, "text": "First item", "label": "positive"},
            {"id": 2, "text": "Second item", "label": "negative"},
            {"id": 3, "text": "Third item", "label": "neutral"},
        ],
        "metadata": {
            "created_at": "2024-01-01",
            "author": "test_user",
        }
    }


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Skip slow tests by default unless --run-slow is passed
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle custom markers."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
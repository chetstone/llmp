import pytest
from unittest.mock import Mock, patch, MagicMock
import json


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.run for llm commands."""
    with patch('subprocess.run') as mock_run:
        # Default successful response
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Test response"
        mock_run.return_value.stderr = ""
        yield mock_run


@pytest.fixture
def mock_stdin():
    """Mock stdin for testing piped input."""
    with patch('sys.stdin') as mock:
        mock.isatty.return_value = True  # Default to terminal mode
        yield mock


@pytest.fixture
def mock_open_tty():
    """Mock opening /dev/tty for terminal input."""
    with patch('builtins.open') as mock_open:
        yield mock_open

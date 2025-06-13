# tests/conftest.py
import pytest
from unittest.mock import Mock, MagicMock
import sys

@pytest.fixture
def mock_llm():
    """Mock the llm module for testing."""
    mock = MagicMock()

    # Mock the database
    mock_db = Mock()
    mock_conversation = Mock()
    mock_conversation.id = "test-conv-123"
    mock_conversation.responses = []

    mock_db.get_conversation.return_value = mock_conversation
    mock_db.get_all_conversations.return_value = [mock_conversation]

    mock.get_default_db.return_value = mock_db

    # Mock models
    mock_model = Mock()
    mock_model.model_id = "gpt-4"
    mock_response = Mock()
    mock_response.text.return_value = "Test response"
    mock_response.conversation = mock_conversation
    mock_model.prompt.return_value = mock_response

    mock.get_model.return_value = mock_model
    mock.get_default_model.return_value = mock_model

    # Temporarily replace the module
    sys.modules['llm'] = mock
    yield mock

    # Cleanup
    if 'llm' in sys.modules:
        del sys.modules['llm']

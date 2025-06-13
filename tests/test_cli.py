import pytest
from unittest.mock import Mock, patch
from llmp.cli import LLMPrompt

# tests/test_cli.py
def test_execute_llm(mock_llm):
    """Test LLM execution with mocked llm module."""
    from llmp.cli import LLMPrompt

    app = LLMPrompt()
    response, cid = app._execute_llm("Test prompt", "gpt-4", None)

    assert response == "Test response"
    assert cid == "test-conv-123"

    # Verify the mock was called correctly
    mock_llm.get_model.assert_called_with("gpt-4")

def test_prompt_concatenation():
    """Test that prompts are concatenated correctly."""
    app = LLMPrompt()

    # Test empty prompt
    with patch('builtins.input', side_effect=['!end']):
        with patch.object(app, '_has_stdin_data', return_value=False):
            prompt = app.collect_prompt()
            assert prompt == ""

    # Test terminal only
    with patch('builtins.input', side_effect=['line1', 'line2', '!end']):
        with patch.object(app, '_has_stdin_data', return_value=False):
            prompt = app.collect_prompt()
            assert prompt == "line1\nline2"


def test_model_determination():
    """Test model determination logic."""
    app = LLMPrompt()

    # Mock the llm module
    with patch('llm.get_default_model') as mock_model:
        mock_model.return_value.model_id = 'gpt-4'

        model, conv, info = app._determine_model_and_conversation(
            model=None,
            continue_conv=False,
            cid=None
        )

        assert model == 'gpt-4'
        assert conv is None
        assert info == ""

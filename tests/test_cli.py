import pytest
import json
from unittest.mock import Mock, patch, MagicMock, mock_open
from io import StringIO

from llmp.cli import LLMPrompt


class TestLLMPrompt:
    """Test suite for LLMPrompt class."""

    def test_init_success(self, mock_subprocess):
        """Test successful initialization."""
        app = LLMPrompt()
        # Should check for llm command
        mock_subprocess.assert_called_with(
            ['llm', '--version'],
            capture_output=True,
            check=True
        )

    def test_init_llm_not_found(self, mock_subprocess):
        """Test initialization when llm command not found."""
        mock_subprocess.side_effect = FileNotFoundError()

        with pytest.raises(SystemExit):
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                LLMPrompt()

    def test_run_llm_command(self, mock_subprocess):
        """Test running llm commands."""
        app = LLMPrompt()

        mock_subprocess.return_value.stdout = "output"
        mock_subprocess.return_value.returncode = 0

        output, code = app._run_llm_command(['models', 'default'])

        assert output == "output"
        assert code == 0
        mock_subprocess.assert_called_with(
            ['llm', 'models', 'default'],
            input=None,
            text=True,
            capture_output=True
        )

    def test_get_default_model(self, mock_subprocess):
        """Test getting default model."""
        app = LLMPrompt()

        mock_subprocess.return_value.stdout = "gpt-4\n"
        mock_subprocess.return_value.returncode = 0

        model = app._get_default_model()
        assert model == "gpt-4"

    def test_get_conversation_model(self, mock_subprocess):
        """Test getting model from conversation."""
        app = LLMPrompt()

        # Mock the logs response
        logs = [{"model": "gpt-3.5-turbo", "conversation_id": "abc123"}]
        mock_subprocess.return_value.stdout = json.dumps(logs)
        mock_subprocess.return_value.returncode = 0

        model = app._get_conversation_model("abc123")
        assert model == "gpt-3.5-turbo"

    def test_get_latest_conversation_id(self, mock_subprocess):
        """Test getting latest conversation ID."""
        app = LLMPrompt()

        logs = [{"conversation_id": "xyz789", "model": "gpt-4"}]
        mock_subprocess.return_value.stdout = json.dumps(logs)
        mock_subprocess.return_value.returncode = 0

        cid = app._get_latest_conversation_id()
        assert cid == "xyz789"


class TestPromptCollection:
    """Test prompt collection functionality."""

    def test_collect_prompt_terminal_only(self, mock_subprocess):
        """Test collecting prompt from terminal only."""
        app = LLMPrompt()

        # Mock terminal input
        mock_tty = StringIO("line1\nline2\n!end\n")

        with patch('builtins.open', mock_open(read_data="")) as m:
            m.return_value = mock_tty
            with patch.object(app, '_has_stdin_data', return_value=False):
                prompt = app.collect_prompt()

        assert prompt == "line1\nline2"

    def test_collect_prompt_stdin_only(self, mock_subprocess, mock_stdin):
        """Test collecting prompt from stdin only."""
        app = LLMPrompt()

        # Mock piped input
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "piped input"

        # Mock terminal input (immediate !end)
        mock_tty = StringIO("!end\n")

        with patch('builtins.open', mock_open(read_data="")) as m:
            m.return_value = mock_tty
            with patch.object(app, '_has_stdin_data', return_value=True):
                with patch('select.select', return_value=([mock_stdin], [], [])):
                    prompt = app.collect_prompt()

        assert prompt == "piped input"

    def test_collect_prompt_both_inputs(self, mock_subprocess, mock_stdin):
        """Test collecting prompt from both stdin and terminal."""
        app = LLMPrompt()

        # Mock piped input
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "piped input"

        # Mock terminal input
        mock_tty = StringIO("terminal line\n!end\n")

        with patch('builtins.open', mock_open(read_data="")) as m:
            m.return_value = mock_tty
            with patch.object(app, '_has_stdin_data', return_value=True):
                with patch('select.select', return_value=([mock_stdin], [], [])):
                    prompt = app.collect_prompt()

        assert prompt == "terminal line\npiped input"

    def test_collect_prompt_strips_single_newline(self, mock_subprocess):
        """Test that only a single trailing newline is stripped."""
        app = LLMPrompt()

        # Input: "line1\n\n!end\n"
        # After reading lines: ["line1", ""]  (empty line before !end)
        # After joining: "line1\n"
        # After stripping single newline: "line1"
        mock_tty = StringIO("line1\n\n!end\n")

        with patch('builtins.open', mock_open(read_data="")) as m:
            m.return_value = mock_tty
            with patch.object(app, '_has_stdin_data', return_value=False):
                prompt = app.collect_prompt()

        # The empty line is preserved, but the final trailing newline is stripped
        assert prompt == "line1"

    def test_collect_prompt_preserves_multiple_trailing_newlines(self, mock_subprocess):
        """Test that multiple trailing newlines are preserved (only one stripped)."""
        app = LLMPrompt()

        # Test with multiple newlines in the middle
        mock_tty = StringIO("line1\n\n\nline2\n!end\n")

        with patch('builtins.open', mock_open(read_data="")) as m:
            m.return_value = mock_tty
            with patch.object(app, '_has_stdin_data', return_value=False):
                prompt = app.collect_prompt()

        # Should preserve all internal newlines
        assert prompt == "line1\n\n\nline2"


class TestModelDetermination:
    """Test model determination logic."""

    def test_explicit_model(self, mock_subprocess):
        """Test when model is explicitly specified."""
        app = LLMPrompt()

        model, cid, info = app._determine_model_and_conversation(
            model="gpt-4",
            continue_conv=False,
            cid=None
        )

        assert model == "gpt-4"
        assert cid is None
        assert info == ""

    def test_continue_with_new_model(self, mock_subprocess):
        """Test continuing conversation with new model."""
        app = LLMPrompt()

        model, cid, info = app._determine_model_and_conversation(
            model="gpt-4",
            continue_conv=True,
            cid="abc123"
        )

        assert model == "gpt-4"
        assert cid == "abc123"
        assert info == " (forcing model on CID: abc123)"

    def test_continue_previous_conversation(self, mock_subprocess):
        """Test continuing previous conversation without CID."""
        app = LLMPrompt()

        model, cid, info = app._determine_model_and_conversation(
            model=None,
            continue_conv=True,
            cid=None
        )

        assert model == "(continuing previous conversation)"
        assert cid is None
        assert info == " (using last conversation)"

    def test_default_model(self, mock_subprocess):
        """Test using default model."""
        app = LLMPrompt()

        mock_subprocess.return_value.stdout = "claude-3-opus\n"
        mock_subprocess.return_value.returncode = 0

        model, cid, info = app._determine_model_and_conversation(
            model=None,
            continue_conv=False,
            cid=None
        )

        assert model == "claude-3-opus"
        assert cid is None
        assert info == ""


class TestLLMExecution:
    """Test LLM execution."""

    def test_execute_llm_new_conversation(self, mock_subprocess):
        """Test executing LLM for new conversation."""
        app = LLMPrompt()

        # First call is for execution
        mock_subprocess.return_value.stdout = "This is the response"
        mock_subprocess.return_value.returncode = 0

        # Mock getting the conversation ID
        with patch.object(app, '_get_latest_conversation_id', return_value='new-conv-123'):
            response, cid = app._execute_llm(
                "Test prompt",
                "gpt-4",
                continue_conv=False,
                conversation_id=None
            )

        assert response == "This is the response"
        assert cid == "new-conv-123"

        # Check the command that was called
        call_args = mock_subprocess.call_args_list[-1]
        assert call_args[0][0] == ['llm', '-m', 'gpt-4', 'Test prompt']

    def test_execute_llm_continue_conversation(self, mock_subprocess):
        """Test continuing a conversation."""
        app = LLMPrompt()

        mock_subprocess.return_value.stdout = "Continued response"
        mock_subprocess.return_value.returncode = 0

        with patch.object(app, '_get_latest_conversation_id', return_value='abc123'):
            response, cid = app._execute_llm(
                "Continue this",
                None,
                continue_conv=True,
                conversation_id=None
            )

        assert response == "Continued response"
        assert cid == "abc123"

        # Check the command
        call_args = mock_subprocess.call_args_list[-1]
        assert call_args[0][0] == ['llm', '-c', 'Continue this']

    def test_execute_llm_with_cid(self, mock_subprocess):
        """Test continuing specific conversation."""
        app = LLMPrompt()

        mock_subprocess.return_value.stdout = "Response for specific conversation"
        mock_subprocess.return_value.returncode = 0

        with patch.object(app, '_get_latest_conversation_id', return_value='xyz789'):
            response, cid = app._execute_llm(
                "Test prompt",
                "gpt-4",
                continue_conv=False,
                conversation_id="xyz789"
            )

        assert response == "Response for specific conversation"

        # Check the command
        call_args = mock_subprocess.call_args_list[-1]
        assert call_args[0][0] == ['llm', '-m', 'gpt-4', '--cid', 'xyz789', 'Test prompt']

    def test_execute_llm_error(self, mock_subprocess):
        """Test handling LLM execution errors."""
        app = LLMPrompt()

        mock_subprocess.return_value.stdout = ""
        mock_subprocess.return_value.stderr = "Error: Invalid model"
        mock_subprocess.return_value.returncode = 1

        with pytest.raises(RuntimeError, match="Invalid model"):
            app._execute_llm("Test", "bad-model", False, None)


class TestClipboard:
    """Test clipboard functionality."""

    @pytest.mark.skipif(not pytest.importorskip("pyperclip"),
                        reason="pyperclip not installed")
    def test_copy_to_clipboard_success(self, mock_subprocess):
        """Test successful clipboard copy."""
        app = LLMPrompt()

        with patch('pyperclip.copy') as mock_copy:
            result = app._copy_to_clipboard(
                "Test prompt",
                "Test response",
                "gpt-4",
                "conv-123"
            )

        assert result is True

        # Check clipboard content
        clipboard_content = mock_copy.call_args[0][0]
        assert "## Prompt:\nTest prompt" in clipboard_content
        assert "#### Model: gpt-4" in clipboard_content
        assert "#### Conversation: conv-123" in clipboard_content
        assert "## Response:\nTest response" in clipboard_content

    def test_copy_to_clipboard_no_pyperclip(self, mock_subprocess):
        """Test clipboard when pyperclip not available."""
        app = LLMPrompt()

        with patch('llmp.cli.HAS_CLIPBOARD', False):
            result = app._copy_to_clipboard("prompt", "response", "model", None)

        assert result is False


class TestGitHub:
    """Test GitHub integration."""

    def test_parse_github_spec_full(self, mock_subprocess):
        """Test parsing full GitHub specification."""
        app = LLMPrompt()

        owner, repo, issue = app._parse_github_spec("microsoft/vscode#123")
        assert owner == "microsoft"
        assert repo == "vscode"
        assert issue == 123

    def test_parse_github_spec_default_owner(self, mock_subprocess):
        """Test parsing with default owner."""
        app = LLMPrompt()

        owner, repo, issue = app._parse_github_spec("my-repo#456")
        assert owner == "chetstone"
        assert repo == "my-repo"
        assert issue == 456

    def test_parse_github_spec_new_issue_full(self, mock_subprocess):
        """Test parsing for new issue with full spec."""
        app = LLMPrompt()

        owner, repo, issue = app._parse_github_spec("microsoft/vscode")
        assert owner == "microsoft"
        assert repo == "vscode"
        assert issue is None

    def test_parse_github_spec_new_issue_default_owner(self, mock_subprocess):
        """Test parsing for new issue with default owner."""
        app = LLMPrompt()

        owner, repo, issue = app._parse_github_spec("my-repo")
        assert owner == "chetstone"
        assert repo == "my-repo"
        assert issue is None

    def test_truncate_for_title(self, mock_subprocess):
        """Test title truncation."""
        app = LLMPrompt()

        # Short prompt
        assert app._truncate_for_title("Short prompt") == "Short prompt"

        # Long prompt
        long_text = "This is a very long prompt that exceeds the maximum length allowed for a GitHub issue title and should be truncated"
        truncated = app._truncate_for_title(long_text)
        assert len(truncated) == 80
        assert truncated.endswith("...")

        # Multi-line prompt (should use first line only)
        multi_line = "First line\nSecond line\nThird line"
        assert app._truncate_for_title(multi_line) == "First line"

    @pytest.mark.skipif(not pytest.importorskip("github"),
                        reason="PyGithub not installed")
    def test_post_to_github_success(self, mock_subprocess, monkeypatch):
        """Test successful GitHub posting."""
        app = LLMPrompt()

        # Set GitHub token
        monkeypatch.setenv('GITHUB_TOKEN', 'fake-token')

        # Mock GitHub objects
        mock_issue = Mock()
        mock_repo = Mock()
        mock_repo.get_issue.return_value = mock_issue
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo

        # Patch at the module level where it's imported
        with patch('llmp.cli.Github', return_value=mock_github):
            result = app._post_to_github(
                "Test prompt",
                "Test response",
                "gpt-4",
                "conv-123",
                "owner/repo#42"
            )

        assert result is True
        mock_github.get_repo.assert_called_with("owner/repo")
        mock_repo.get_issue.assert_called_with(42)

        # Check comment content
        comment = mock_issue.create_comment.call_args[0][0]
        assert "## Prompt:\nTest prompt" in comment
        assert "#### Model: gpt-4" in comment
        assert "## Response:\nTest response" in comment

    @pytest.mark.skipif(not pytest.importorskip("github"),
                        reason="PyGithub not installed")
    def test_post_to_github_new_issue(self, mock_subprocess, monkeypatch):
        """Test creating new GitHub issue."""
        app = LLMPrompt()

        monkeypatch.setenv('GITHUB_TOKEN', 'fake-token')

        # Mock GitHub objects
        mock_issue = Mock()
        mock_issue.number = 789
        mock_repo = Mock()
        mock_repo.create_issue.return_value = mock_issue
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo

        with patch('llmp.cli.Github', return_value=mock_github):
            result = app._post_to_github(
                "Test prompt for new issue",
                "Test response",
                "gpt-4",
                "conv-123",
                "my-repo"  # No issue number
            )

        assert result is True
        mock_github.get_repo.assert_called_with("chetstone/my-repo")
        mock_repo.create_issue.assert_called_once()

        # Check the issue creation call
        call_args = mock_repo.create_issue.call_args
        assert call_args[1]['title'] == "Test prompt for new issue"
        assert "## Prompt:\nTest prompt for new issue" in call_args[1]['body']

    def test_post_to_github_no_token(self, mock_subprocess, monkeypatch):
        """Test GitHub posting without token."""
        app = LLMPrompt()

        monkeypatch.delenv('GITHUB_TOKEN', raising=False)

        result = app._post_to_github(
            "prompt", "response", "model", None, "owner/repo#1"
        )

        assert result is False

    def test_post_to_github_invalid_format(self, mock_subprocess):
        """Test GitHub posting with invalid issue format."""
        app = LLMPrompt()

        result = app._post_to_github(
            "prompt", "response", "model", None, "invalid#format#extra"
        )

        assert result is False


class TestMainIntegration:
    """Test main integration flow."""

    def test_full_flow_new_conversation(self, mock_subprocess):
        """Test full flow for new conversation."""
        app = LLMPrompt()

        # Mock prompt collection
        with patch.object(app, 'collect_prompt', return_value="What is 2+2?"):
            # Mock LLM execution
            with patch.object(app, '_execute_llm', return_value=("4", "new-conv")):
                # Mock clipboard
                with patch.object(app, '_copy_to_clipboard', return_value=True):
                    result = app.run(
                        model="gpt-4",
                        continue_conv=False,
                        cid=None,
                        clipboard=True,
                        github_issue=None
                    )

        assert result == 0

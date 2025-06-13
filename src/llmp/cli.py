#!/usr/bin/env python3
"""
LLM prompt tool with clipboard and GitHub integration.
"""

import sys
import os
import json
import select
from typing import Optional, Tuple, List
import subprocess
from pathlib import Path

import click
try:
    import pyperclip
    HAS_CLIPBOARD = True
except ImportError:
    HAS_CLIPBOARD = False

try:
    from github import Github
    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False


# First, we need to find and import the llm module from the uv tool installation
def find_llm_module():
    """Find and import the llm module from uv tools."""
    # Try to find llm installation via uv
    try:
        result = subprocess.run(
            ["uv", "tool", "dir"],
            capture_output=True,
            text=True,
            check=True
        )
        tool_dir = Path(result.stdout.strip())
        llm_dir = tool_dir / "llm"

        if llm_dir.exists():
            # Find the site-packages directory
            for path in llm_dir.rglob("site-packages"):
                if path.is_dir():
                    sys.path.insert(0, str(path))
                    try:
                        import llm
                        return llm
                    except ImportError:
                        continue

        # Fallback: try to import directly (might work if in same env)
        import llm
        return llm

    except Exception as e:
        print(f"Error: Could not find llm module: {e}", file=sys.stderr)
        print("Make sure llm is installed with: uv tool install llm", file=sys.stderr)
        sys.exit(1)


# Import llm from the uv tool installation
llm = find_llm_module()


class LLMPrompt:
    def __init__(self):
        self.db = llm.get_default_db()

    def _has_stdin_data(self) -> bool:
        """Check if there's data available on stdin without blocking."""
        if sys.stdin.isatty():
            return False

        if hasattr(select, 'select'):
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            return bool(ready)
        else:
            return not sys.stdin.isatty()

    def _read_from_terminal(self) -> List[str]:
        """Read input from terminal (tty) even when stdin is piped."""
        lines = []

        try:
            with open('/dev/tty', 'r') as tty:
                while True:
                    try:
                        line = tty.readline()
                        if not line:
                            break
                        line = line.rstrip('\n')
                        if line == "!end":
                            break
                        lines.append(line)
                    except KeyboardInterrupt:
                        print("\nInterrupted", file=sys.stderr)
                        break
        except (IOError, OSError):
            # Fallback for Windows or when /dev/tty is not available
            print("Warning: Cannot open /dev/tty, falling back to standard input", file=sys.stderr)
            while True:
                try:
                    line = input()
                    if line == "!end":
                        break
                    lines.append(line)
                except (EOFError, KeyboardInterrupt):
                    break

        return lines

    def collect_prompt(self) -> str:
        """Collect prompt from both piped stdin and terminal input."""
        print("Enter your prompt. Type '!end' at the beginning of a line to finish.")

        piped_input = ""
        if self._has_stdin_data():
            print("(Reading from pipe...)")
            piped_input = sys.stdin.read()

        terminal_lines = self._read_from_terminal()
        terminal_input = '\n'.join(terminal_lines) if terminal_lines else ""

        if piped_input and terminal_input:
            prompt = terminal_input + '\n' + piped_input
        elif piped_input:
            prompt = piped_input
        elif terminal_input:
            prompt = terminal_input
        else:
            prompt = ""

        if prompt and prompt.endswith('\n'):
            prompt = prompt[:-1]

        return prompt

    def _determine_model_and_conversation(self, model: Optional[str], continue_conv: bool,
                                          cid: Optional[str]) -> Tuple[str, Optional[llm.Conversation], str]:
        """Determine the model name, conversation, and display information."""
        display_model = ""
        display_cid_info = ""
        conversation = None

        if continue_conv or cid:
            if model:
                display_model = model
                if cid:
                    display_cid_info = f" (forcing model on CID: {cid})"
                    conversation = self.db.get_conversation(cid)
                else:
                    display_cid_info = " (forcing model on previous conversation)"
                    # Get the last conversation
                    conversations = list(self.db.get_all_conversations())
                    if conversations:
                        conversation = conversations[0]

            elif cid:
                display_cid_info = f" (CID: {cid})"
                conversation = self.db.get_conversation(cid)

                if conversation and conversation.responses:
                    display_model = conversation.responses[0].model.model_id
                else:
                    display_model = "(unknown original model)"

            else:
                display_model = "(continuing previous conversation)"
                display_cid_info = " (using last conversation)"
                conversations = list(self.db.get_all_conversations())
                if conversations:
                    conversation = conversations[0]
                    if conversation.responses:
                        display_model = conversation.responses[0].model.model_id

        else:
            if model:
                display_model = model
            else:
                try:
                    default_model = llm.get_default_model()
                    display_model = default_model.model_id
                except Exception:
                    display_model = "(unknown default model)"

        return display_model, conversation, display_cid_info

    def _execute_llm(self, prompt: str, model_name: Optional[str],
                     conversation: Optional[llm.Conversation]) -> Tuple[str, str]:
        """Execute the LLM with the given prompt."""
        try:
            if conversation:
                # Continue existing conversation
                if model_name and not model_name.startswith("("):
                    model = llm.get_model(model_name)
                else:
                    # Use the model from the conversation
                    model = None

                response = conversation.prompt(prompt, model=model)

            else:
                # New conversation
                if model_name and not model_name.startswith("("):
                    model = llm.get_model(model_name)
                else:
                    model = llm.get_default_model()

                response = model.prompt(prompt, log=True)

            # Get the conversation ID
            if hasattr(response, 'conversation') and response.conversation:
                actual_cid = response.conversation.id
            else:
                # Get the latest conversation
                conversations = list(self.db.get_all_conversations())
                actual_cid = conversations[0].id if conversations else None

            return response.text(), actual_cid

        except Exception as e:
            raise RuntimeError(f"LLM execution failed: {e}")

    def _copy_to_clipboard(self, prompt: str, response: str, model: str, cid: Optional[str]) -> bool:
        """Copy formatted output to clipboard."""
        if not HAS_CLIPBOARD:
            print("Warning: pyperclip not installed. Cannot copy to clipboard.", file=sys.stderr)
            return False

        cid_line = f"\n#### Conversation: {cid}" if cid else ""

        clipboard_text = f"""## Prompt:
{prompt}

#### Model: {model}{cid_line}

## Response:
{response}"""

        try:
            pyperclip.copy(clipboard_text)
            if cid:
                print(f"(Prompt/Response (CID: {cid}) copied to clipboard)")
            else:
                print("(Prompt/Response copied to clipboard - CID retrieval failed)")
            return True
        except Exception as e:
            print(f"Warning: Failed to copy to clipboard: {e}", file=sys.stderr)
            return False

    def _post_to_github(self, prompt: str, response: str, model: str, cid: Optional[str],
                        issue_spec: str) -> bool:
        """Post formatted output as a GitHub issue comment."""
        if not HAS_GITHUB:
            print("Error: PyGithub not installed. Install with: pip install PyGithub", file=sys.stderr)
            return False

        try:
            repo_part, issue_number = issue_spec.split('#')
            owner, repo_name = repo_part.split('/')
            issue_number = int(issue_number)
        except ValueError:
            print(f"Error: Invalid issue format '{issue_spec}'. Use: owner/repo#number", file=sys.stderr)
            return False

        token = os.environ.get('GITHUB_TOKEN')
        if not token:
            print("Error: GITHUB_TOKEN environment variable not set", file=sys.stderr)
            return False

        try:
            g = Github(token)
            repo = g.get_repo(f"{owner}/{repo_name}")
            issue = repo.get_issue(issue_number)

            cid_line = f"\n#### Conversation: {cid}" if cid else ""
            comment_body = f"""## Prompt:
{prompt}

#### Model: {model}{cid_line}

## Response:
{response}"""

            issue.create_comment(comment_body)
            print(f"Posted to GitHub issue: {issue_spec}")
            return True

        except Exception as e:
            print(f"Error posting to GitHub: {e}", file=sys.stderr)
            return False

    def run(self, model: Optional[str], continue_conv: bool, cid: Optional[str],
            clipboard: bool, github_issue: Optional[str]) -> int:
        """Main execution logic."""

        display_model, conversation, display_cid_info = self._determine_model_and_conversation(
            model, continue_conv, cid
        )

        print(f"Model: {display_model}{display_cid_info}")

        prompt = self.collect_prompt()
        if not prompt:
            print("No prompt provided. Exiting.", file=sys.stderr)
            return 1

        print("---")
        try:
            response, actual_cid = self._execute_llm(prompt, model, conversation)
            print(response)
            print("---")

        except Exception as e:
            print("---")
            print(f"Error: {e}", file=sys.stderr)
            return 1

        if clipboard:
            self._copy_to_clipboard(prompt, response, display_model, actual_cid)

        if github_issue:
            self._post_to_github(prompt, response, display_model, actual_cid, github_issue)

        return 0


@click.command()
@click.option('-C', '--clipboard', is_flag=True, help='Copy prompt and response to clipboard')
@click.option('-m', '--model', help='Specify model to use')
@click.option('-c', '--continue', 'continue_conv', is_flag=True, help='Continue previous conversation')
@click.option('--cid', help='Continue specific conversation by ID')
@click.option('--github-issue', help='Post to GitHub issue (format: owner/repo#number)')
@click.pass_context
def main(ctx, clipboard, model, continue_conv, cid, github_issue):
    """LLM prompt tool with clipboard and GitHub integration."""
    # Handle --cid implying continuation
    if cid:
        continue_conv = True

    app = LLMPrompt()
    sys.exit(app.run(model, continue_conv, cid, clipboard, github_issue))


if __name__ == '__main__':
    main()

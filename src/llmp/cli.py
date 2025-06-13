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


class LLMPrompt:
    def __init__(self):
        # Verify llm command exists
        try:
            subprocess.run(['llm', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: 'llm' command not found. Please install with: uv tool install llm",
                  file=sys.stderr)
            sys.exit(1)

    def _run_llm_command(self, args: List[str], input_text: Optional[str] = None) -> Tuple[str, int]:
        """Run llm command and return (output, exit_code)."""
        try:
            result = subprocess.run(
                ['llm'] + args,
                input=input_text,
                text=True,
                capture_output=True
            )
            return result.stdout, result.returncode
        except Exception as e:
            return str(e), 1

    def _get_conversation_model(self, cid: str) -> Optional[str]:
        """Get the model used in a conversation."""
        output, code = self._run_llm_command(['logs', '--cid', cid, '--json'])
        if code == 0 and output:
            try:
                logs = json.loads(output)
                if logs and len(logs) > 0:
                    return logs[0].get('model')
            except json.JSONDecodeError:
                pass
        return None

    def _get_default_model(self) -> str:
        """Get the default model."""
        output, code = self._run_llm_command(['models', 'default'])
        if code == 0 and output.strip():
            return output.strip()
        return "(unknown default model)"

    def _get_latest_conversation_id(self) -> Optional[str]:
        """Get the most recent conversation ID."""
        output, code = self._run_llm_command(['logs', '-n', '1', '--json'])
        if code == 0 and output:
            try:
                logs = json.loads(output)
                if logs and len(logs) > 0:
                    return logs[0].get('conversation_id')
            except json.JSONDecodeError:
                pass
        return None

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
                                          cid: Optional[str]) -> Tuple[str, Optional[str], str]:
        """Determine the model name, conversation ID, and display information."""
        display_model = ""
        display_cid_info = ""
        conversation_id = cid  # Start with explicit CID if provided

        if continue_conv or cid:
            if model:
                display_model = model
                if cid:
                    display_cid_info = f" (forcing model on CID: {cid})"
                else:
                    display_cid_info = " (forcing model on previous conversation)"
                    # We'll let llm handle finding the last conversation

            elif cid:
                display_cid_info = f" (CID: {cid})"
                # Get model from the conversation
                conv_model = self._get_conversation_model(cid)
                if conv_model:
                    display_model = conv_model
                else:
                    print(f"Warning: Could not determine original model from conversation '{cid}'",
                          file=sys.stderr)
                    display_model = "(unknown original model)"

            else:
                # Just -c flag
                display_model = "(continuing previous conversation)"
                display_cid_info = " (using last conversation)"

        else:
            if model:
                display_model = model
            else:
                display_model = self._get_default_model()

        return display_model, conversation_id, display_cid_info

    def _execute_llm(self, prompt: str, model_name: Optional[str],
                     continue_conv: bool, conversation_id: Optional[str]) -> Tuple[str, Optional[str]]:
        """Execute the LLM with the given prompt."""
        try:
            # Build the command - use 'llm' not 'llm prompt'
            cmd = ["llm"]

            if model_name and not model_name.startswith("("):
                cmd.extend(["-m", model_name])

            if conversation_id:
                cmd.extend(["--cid", conversation_id])
            elif continue_conv:
                cmd.append("-c")

            # Add the prompt as the last argument
            cmd.append(prompt)

            # Execute
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise RuntimeError(f"llm command failed: {error_msg}")

            response = result.stdout.strip()

            # Get the actual conversation ID from the last log entry
            actual_cid = self._get_latest_conversation_id()

            return response, actual_cid

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

    def _parse_github_spec(self, spec: str) -> Tuple[str, str, Optional[int]]:
        """
        Parse GitHub specification.

        Formats:
        - owner/repo#123 -> (owner, repo, 123)
        - repo#123 -> (chetstone, repo, 123)
        - owner/repo -> (owner, repo, None)
        - repo -> (chetstone, repo, None)

        Returns: (owner, repo, issue_number)
        """
        # Split by # if present
        if '#' in spec:
            repo_part, issue_str = spec.split('#', 1)
            try:
                issue_number = int(issue_str)
            except ValueError:
                raise ValueError(f"Invalid issue number: {issue_str}")
        else:
            repo_part = spec
            issue_number = None

        # Split owner/repo
        if '/' in repo_part:
            owner, repo = repo_part.split('/', 1)
        else:
            owner = "chetstone"
            repo = repo_part

        return owner, repo, issue_number

    def _truncate_for_title(self, text: str, max_length: int = 80) -> str:
        """Truncate text for use as issue title."""
        # Take first line only
        first_line = text.split('\n')[0].strip()

        if len(first_line) <= max_length:
            return first_line

        # Truncate and add ellipsis
        return first_line[:max_length-3] + "..."

    def _post_to_github(self, prompt: str, response: str, model: str, cid: Optional[str],
                        issue_spec: str) -> bool:
        """Post formatted output as a GitHub issue comment or create new issue."""
        if not HAS_GITHUB:
            print("Error: PyGithub not installed. Install with: pip install PyGithub", file=sys.stderr)
            return False

        try:
            owner, repo_name, issue_number = self._parse_github_spec(issue_spec)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return False

        token = os.environ.get('GITHUB_TOKEN')
        if not token:
            print("Error: GITHUB_TOKEN environment variable not set", file=sys.stderr)
            return False

        try:
            g = Github(token)
            repo = g.get_repo(f"{owner}/{repo_name}")

            cid_line = f"\n#### Conversation: {cid}" if cid else ""
            body = f"""## Prompt:
{prompt}

#### Model: {model}{cid_line}

## Response:
{response}"""

            if issue_number:
                # Add comment to existing issue
                issue = repo.get_issue(issue_number)
                issue.create_comment(body)
                print(f"Posted to GitHub issue: {owner}/{repo_name}#{issue_number}")
            else:
                # Create new issue
                title = self._truncate_for_title(prompt)
                issue = repo.create_issue(title=title, body=body)
                print(f"Created GitHub issue: {owner}/{repo_name}#{issue.number}")

            return True

        except Exception as e:
            print(f"Error posting to GitHub: {e}", file=sys.stderr)
            return False

    def run(self, model: Optional[str], continue_conv: bool, cid: Optional[str],
            clipboard: bool, github_issue: Optional[str]) -> int:
        """Main execution logic."""

        display_model, conversation_id, display_cid_info = self._determine_model_and_conversation(
            model, continue_conv, cid
        )

        print(f"Model: {display_model}{display_cid_info}")

        prompt = self.collect_prompt()
        if not prompt:
            print("No prompt provided. Exiting.", file=sys.stderr)
            return 1

        print("---")
        try:
            response, actual_cid = self._execute_llm(prompt, model, continue_conv, conversation_id)
            print(response)
            print("---")

        except Exception as e:
            print("---")
            print(f"Error: {e}", file=sys.stderr)
            return 1

        # Update display_model if it was ambiguous and we can get it from logs
        if display_model in ["(continuing previous conversation)", "(unknown default model)"] and actual_cid:
            # Try to get the actual model from the latest log
            output, code = self._run_llm_command(['logs', '-n', '1', '--json'])
            if code == 0 and output:
                try:
                    logs = json.loads(output)
                    if logs and len(logs) > 0:
                        actual_model = logs[0].get('model')
                        if actual_model:
                            display_model = actual_model
                except json.JSONDecodeError:
                    pass

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
@click.option('-g', '--github-issue', help='Post to GitHub (repo, repo#num, owner/repo, owner/repo#num)')
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

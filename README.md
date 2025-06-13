# llmp
Use llm without worrying about shell quoting rules when typing the prompt

## Development Setup
# 1. Create and activate virtual environment
cd llmp
uv venv
source .venv/bin/activate

# 2. Install llmp in editable mode with its dependencies
uv pip install -e .
# Install with all features for development
uv pip install -e ".[dev]"

# This installs:
# - pyperclip (in venv)
# - PyGithub (in venv)
# - click (in venv)
# - llmp itself in editable mode
# But NOT llm - we use the global one

# 3. Install dev dependencies
uv pip install pytest pytest-cov pytest-mock

# 4. Verify it works
llmp --help  # Should work, using global llm

## Project Structure with testing
```
llmp/
├── pyproject.toml
├── src/
│   └── llmp/
│       ├── __init__.py      # ← Note double underscores!
│       ├── __main__.py      # Optional, for python -m llmp
│       └── cli.py
├── tests/
│   ├── __init__.py
│   └── test_cli.py
├── dev.py                   # Optional development runner
└── README.md
```

# Run all tests
pytest

# Run with coverage
pytest --cov=llmp --cov-report=html

# Run specific test file
pytest tests/test_cli.py

# Run specific test
pytest tests/test_cli.py::TestLLMExecution::test_execute_llm_new_conversation

# Run with verbose output
pytest -v

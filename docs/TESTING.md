# Zelyr Testing & CI Baseline

This document outlines the strict testing conventions, environment isolation protocols, and execution boundaries for the Zelyr V11+ codebase. 

## Philosophy

Zelyr interacts with the host desktop (Vision, Mouse, Keyboard, Filesystem). Testing these components blindly leads to flakiness, corrupted developer environments, and false-positive test passes. We enforce a zero-trust execution sandbox for our tests.

## 1. Pytest Markers

All tests **must** be explicitly marked in `pyproject.toml`. Unmarked tests or improperly marked tests will fail CI constraints.

- `@pytest.mark.unit`: Fast, purely isolated tests. No filesystem mutations, no external API calls, no LLM calls. Everything external is mocked.
- `@pytest.mark.integration`: Tests interactions between components (e.g., `ToolRegistry` calling `SandboxRunner`). These may use temporary filesystems.
- `@pytest.mark.e2e`: End-to-end critical path tests (e.g., `tests/integration/test_critical_path.py`). These verify the system from the FastAPI edge to the command execution sandbox.
- `@pytest.mark.gui`: PySide6 GUI tests running in headless mode. These use Qt bot and require explicit teardowns.

## 2. Execution Boundaries & Isolation

### Network & LLM Blocking (`block_external_io`)
By default, the `block_external_io` autouse fixture in `tests/conftest.py` **brutally blocks** any outgoing HTTP requests (`httpx`) and `litellm` calls. Tests that intentionally trigger external IO will raise a `RuntimeError`. If you need to test network interactions, you must either write a mock, or explicit test isolation. 

### Filesystem Isolation (`mock_home_directory`)
Tests should **never** write to the developer's real `~/.zelyr` state. The `mock_home_directory` autouse fixture explicitly overrides `$HOME`, `$XDG_CONFIG_HOME`, `$XDG_DATA_HOME`, and `Path.home()` to redirect all state storage to a temporary directory.

### GUI Teardowns (`robust_teardown_qt_widgets`)
Headless Qt widgets often leak memory or crash `_Py_Dealloc` during Python GC. We use a global `robust_teardown_qt_widgets` fixture to call `.deleteLater()` and flush the Qt event loop (`processEvents()`) to prevent segfaults.

## 3. Running Tests

To run the full suite:
```bash
QT_QPA_PLATFORM=offscreen uv run pytest
```
*Note: `QT_QPA_PLATFORM=offscreen` is mandatory for running headless GUI tests in CI.*

To run unit tests only:
```bash
QT_QPA_PLATFORM=offscreen uv run pytest -m unit
```

To run the master critical path test:
```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/integration/test_critical_path.py -v
```

## 4. The Critical Path Test
The `test_critical_path_websocket_to_sandbox` test is the gold standard of Zelyr's functionality. It spawns the FastAPI server, connects to the Swarm WebSocket, injects a mocked LLM response, and verifies that the `SandboxRunner` executes the tool payload (using Bubblewrap containment) and returns the output successfully through the full pipeline.

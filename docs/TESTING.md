# Zelyr Testing & Headless Verification Guide

This document summarizes the active testing configuration, isolation fixtures, and offline verification workflow configured in [`pyproject.toml`](../pyproject.toml) and [`tests/conftest.py`](../tests/conftest.py).

For detailed developer setup and packaging workflows, see [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).

---

## 1. Consolidated Test Layout & Markers

Active automated tests live directly in top-level [`tests/test_*.py`](../tests/) modules. Legacy subdirectories (`tests/fixtures`, `tests/unit`, `tests/gui`, `tests/e2e`, `tests/integration`, `tests/optional`, `tests/simulation`, `zelyr/tests`) are excluded by `norecursedirs` and `addopts` in `pyproject.toml`, and `addopts` applies `-m 'not e2e and not requires_wayland and not hardware'` by default.

Custom markers declared in `pyproject.toml`:
- `unit`: Fast, purely isolated offline tests.
- `integration`: Component interaction tests using temporary directories/SQLite databases.
- `e2e`: End-to-end flows excluded from default offline CI runs.
- `gui`: Headless Qt / PySide6 widget tests (require `QT_QPA_PLATFORM=offscreen`).
- `requires_wayland` / `hardware`: Host compositor or physical device tests excluded by default.

---

## 2. Automatic Headless Isolation (`tests/conftest.py`)

To ensure tests never mutate the developer's `~/.config/zelyr/` directory or trigger external inference/network calls:
1. **Offscreen Qt Rendering:** Sets `QT_QPA_PLATFORM=offscreen`.
2. **Home Directory Isolation:** Redirects `$HOME`, `$USERPROFILE`, `$XDG_CONFIG_HOME`, `$XDG_DATA_HOME`, and `Path.home()` to a per-test `TemporaryDirectory`.
3. **Outbound Network Blocking:** Monkeypatches `httpx.get`, `httpx.post`, `urllib.request.urlopen`, and `litellm.completion` so accidental unmocked external IO fails immediately (`RuntimeError`).
4. **No Desktop Popups:** Subprocess/browser launcher calls (`webbrowser.open`, `xdg-open`) must be mocked cleanly.

---

## 3. Running Tests & Static Checks

```bash
# Run the consolidated offline test suite
QT_QPA_PLATFORM=offscreen ZELYR_TESTING=1 uv run pytest -v

# Run targeted subsystem / boundary suites
QT_QPA_PLATFORM=offscreen ZELYR_TESTING=1 uv run pytest -v \
  tests/test_execution_policy_management.py \
  tests/test_execution_lifecycle_diagnostics.py \
  tests/test_execution_observability.py \
  tests/test_configuration_diagnostics.py \
  tests/test_model_compatibility_diagnostics.py \
  tests/test_execution_failure_boundaries.py

# Static linting and diff whitespace checks
uv run ruff check .
git diff --check
```

> **Note:** On loaded machines, `tests/test_model_matrix.py::TestModelMatrixEvaluator::test_mock_matrix_evaluation_speed_and_determinism` enforces a strict 1.0s wall-clock ceiling (`elapsed < 1.0`) and may fail under high background CPU load without indicating a logic regression.

# AXIOM Developer Workflow & Verification Guide

This guide details the development environment setup, testing standards, linting procedures, and packaging workflows for **AXIOM** (v1.0.0).

---

## 1. Environment Setup & Toolchain `[VERIFIED]`

### Prerequisites:
- **Operating System:** Linux (POSIX compliant; Wayland/Hyprland recommended for runtime compositor features).
- **Python:** Supported version range: `>= 3.10, < 3.12` (Python 3.11.15 is active in the repository `.venv`).
- **Package Manager:** [`uv`](https://docs.astral.sh/uv/) is the designated local toolchain manager.

### Virtual Environment Initialization:
```bash
# Clone the repository
git clone https://github.com/Rutytoi220/Axiom.git
cd Axiom

# Sync virtual environment using uv
uv sync

# Activate environment (optional when using 'uv run')
source .venv/bin/activate
```

---

## 2. Testing & Verification Standards `[VERIFIED]`

### A. The Core Verification Command
Every code modification must be verified using the automated test battery:
```bash
uv run pytest -v
```

### B. Test Topology & Ignored Subdirectories
AXIOM's test configuration is declared in [`pyproject.toml`](../pyproject.toml):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
norecursedirs = [
    "tests/fixtures", "tests/unit", "tests/gui", "tests/e2e",
    "tests/integration", "tests/optional", "tests/simulation",
    "axiom/tests", "dist", "build", ".venv"
]
addopts = "--ignore=tests/fixtures --ignore=tests/unit --ignore=tests/gui --ignore=tests/e2e --ignore=tests/integration --ignore=tests/optional --ignore=tests/simulation --ignore=axiom/tests -m 'not e2e and not requires_wayland and not hardware'"
```

The active, consolidated test battery consists of 61 tests located directly in `tests/`:
1. `tests/test_context_budget_fix.py`: 8K context token window and tier-aware schema filtering (5 tests).
2. `tests/test_fast_path_router.py`: Hybrid fastest-path routing and desktop CLI interception (8 tests).
3. `tests/test_install_bootstrap.py`: Directory auto-bootstrap, path portability, and entry points (5 tests).
4. `tests/test_multi_turn_payload.py`: Context precedence hierarchy, Turn 2 ReAct, and thinking tokens (4 tests).
5. `tests/test_shell_authorization.py`: Shell authorization confirmation gates and fail-closed rules (9 tests).
6. `tests/test_swarm_blackboard.py`: Swarm blackboard store, namespace isolation, and rollback (17 tests).
7. `tests/test_web_search.py`: SSRF security, Tailscale/CGNAT blocking, and prompt injection containment (13 tests).

### C. Test Isolation & Headless Safeguards ([`tests/conftest.py`](../tests/conftest.py))
To prevent side effects on the developer's workstation during automated testing, `tests/conftest.py` enforces three automatic safeguards:
1. **Offscreen Qt Rendering:** Automatically sets `os.environ["QT_QPA_PLATFORM"] = "offscreen"`.
2. **Home Directory Isolation (`mock_home_directory`):**  
   Autouse fixture that points `HOME`, `USERPROFILE`, `XDG_CONFIG_HOME`, and `XDG_DATA_HOME` to a temporary directory (`tempfile.TemporaryDirectory`). Tests never touch or mutate real `~/.config/` or `~/.local/`.
3. **Outbound Network Blocking (`block_external_io`):**  
   Autouse fixture that monkeypatches `httpx.post`, `httpx.get`, `urllib.request.urlopen`, and `litellm.completion` to raise a `RuntimeError` if an unmocked network request is attempted.
4. **Desktop Process Mocking:**  
   Calls to `webbrowser.open`, `webbrowser.open_new_tab`, `xdg-open`, and external subprocess launchers must be mocked in unit tests (as verified in [`tests/test_fast_path_router.py`](../tests/test_fast_path_router.py)).

---

## 3. Static Analysis & Linting `[VERIFIED]`

Code quality is enforced via **Ruff** configured in [`pyproject.toml`](../pyproject.toml):
```toml
[tool.ruff]
target-version = "py310"
line-length = 120

[tool.ruff.lint]
select = [
    "E9",      # Syntax and structural runtime errors
    "F",       # Pyflakes (undefined variables, invalid syntax)
]
ignore = [
    "F401",    # Unused imports (public package re-exports)
    "F841",    # Unused local variables
    "F541",    # F-string without placeholders
    "F811",    # Redefined function or variable
]
```

### Running the Linter:
```bash
uv run ruff check .
```
All code must pass clean with zero errors before considering any task complete.

---

## 4. Packaging, Distribution & Bootstrapping `[VERIFIED]`

### A. Wheel Package Build
AXIOM is packaged as a standard Python wheel distribution via `setuptools`.

Build through the isolated-source wrapper rather than invoking
`uv build --wheel` or `python -m build --wheel` directly. Setuptools reuses
`build/lib` in the project root, so a module deleted after an earlier build
can be silently re-added to the wheel:

```bash
uv run python scripts/build_dist.py --wheel --no-isolation
```
This produces `dist/axiom_core-1.0.0-py3-none-any.whl`.

The script copies only the build inputs (`pyproject.toml`, metadata files and
the `axiom` package) into a temporary directory, builds there, and verifies
that every `.py` file in the wheel exists in the current source tree.

### B. Clean Installation Smoke Test
To verify the distribution in an isolated environment without polluting the host:
```bash
# Create ephemeral virtualenv
python3 -m venv /tmp/axiom_smoke_env
source /tmp/axiom_smoke_env/bin/activate

# Install the built wheel
pip install dist/axiom_core-1.0.0-py3-none-any.whl

# Verify executable console entry points
axiom --help
axiomd --help

# Teardown
deactivate
rm -rf /tmp/axiom_smoke_env
```

### C. Automatic First-Run Directory Bootstrapping
When AXIOM is executed on a clean machine where `~/.config/axiom/` does not exist, [`axiom.config.initialize_directories()`](../axiom/config.py#L20-L36) automatically creates the required hierarchy:
- `~/.config/axiom/tools.d/` (Dynamic tool plugin drop-in directory)
- `~/.config/axiom/plugins/` (Extension modules)
- `~/.config/axiom/sessions/` (Saved session databases)
- `~/.config/axiom/logs/` (System execution logs)
- `~/.config/axiom/models/` (Local model configuration cache)

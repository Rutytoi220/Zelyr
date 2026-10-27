# Zelyr Tool Creation & Extension Guide

This guide documents the real pattern for adding, configuring, and testing tools in **Zelyr** (v1.0.0).  
Zelyr supports two complementary tool creation models:
1. **Dynamic File Plugins (`~/.config/zelyr/tools.d/<name>.py`):** Standalone Python files automatically loaded and hot-reloaded by [`zelyr/core/plugins.py`](../zelyr/core/plugins.py).
2. **Subclassed Native Tools ([`zelyr/tools/base.py`](../zelyr/tools/base.py)):** Structured object-oriented tool classes inheriting from `BaseTool`.

---

## 1. Tool Capability Tiers & Precedence `[VERIFIED]`

Every tool in Zelyr belongs to a capability tier that controls when it is exposed to the local reasoning model:

| Tier | Category | Token Cost | Target Domain | Example Tools |
| :---: | :--- | :---: | :--- | :--- |
| **0** | **Core System** | 150 | Fundamental shell execution & window discovery | `execute_command`, `manage_desktop_window` |
| **1** | **Deterministic IPC** | 150 | Native Linux IPC (`hyprctl`, `dbus`, `systemctl`, `pactl`) | `manage_system_process`, `manage_system_clipboard`, `query_system_journal`, `manage_workspace_file` |
| **2** | **Browser & Web** | 250 | WebExtension DOM bridge (`ws://127.0.0.1:41144`), web search, page extraction | `interact_with_browser`, `web_search`, `fetch_web_page` |
| **3** | **Visual Fallback** | 600 | Set-of-Mark (SoM) bounding box detection, OCR, VLM | `interact_with_ui`, `capture_som_screen` |

### Security Rings (`REQUIRED_RING`):
- `Ring 0`: System-critical built-in tools. Always accessible by default.
- `Ring 1`: Elevated tools requiring explicit authorization.
- `Ring 2` / `Ring 3`: Sandboxed user tools.

---

## 2. Dynamic Plugin Contract (`tools.d/<name>.py`) `[VERIFIED]`

To create a dynamic tool, place a single Python file into the active tools directory:
`~/.config/zelyr/tools.d/<tool_name>.py` (or `$ZELYR_CONFIG_DIR/tools.d/<tool_name>.py`).

### Required Module Exports:

```python
"""Example Dynamic Tool: inspect_git_status.py"""

import json
import subprocess
from typing import Any, Dict

# 1. OpenAPI-compatible function schema
TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_git_status",
        "description": "Inspect git status and uncommitted changes in a workspace repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to git repository root (defaults to current working directory).",
                },
                "short": {
                    "type": "boolean",
                    "description": "Whether to return short-format status.",
                },
            },
            "required": [],
        },
    },
}

# 2. Interactive TUI spinner message
TUI_HINT: str = "🔍 Checking git repository status..."

# 3. Security ring level (0=system, 1=elevated, 2=user)
REQUIRED_RING: int = 0

# 4. Capability tier (0=Core, 1=IPC/OS, 2=Browser, 3=Vision)
TIER: int = 1
IS_CORE: bool = False
TOKEN_COST: int = 150

# 5. Asynchronous execution handler
async def execute(repo_path: str = ".", short: bool = True, **kwargs) -> str:
    """Execute git status safely and return clean output."""
    cmd = ["git", "status"]
    if short:
        cmd.append("--short")
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        output = proc.stdout.strip() or "(clean working tree)"
        return json.dumps({
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "status": output,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "error": "Git status command timed out after 10 seconds.",
            "remedy_hint": "Verify repository path or check if repository is locked.",
            "allowed_actions": [],
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "remedy_hint": "Check if git is installed and repo_path is valid.",
            "allowed_actions": [],
        })
```

---

## 3. Subclassing `BaseTool` ([`zelyr/tools/core.py`](../zelyr/tools/core.py)) `[VERIFIED]`

For core tools integrated directly into the `zelyr.tools` package, inherit from `BaseTool`:

```python
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

class InspectServiceTool(BaseTool):
    """Inspect systemd unit service status."""

    def __init__(self):
        super().__init__(
            tool_id="inspect_service",
            name="inspect_service",
            description="Query the status of a local systemd service.",
        )
        self.add_parameter(ToolParameter(
            name="service_name",
            type="string",
            description="Name of the systemd unit (e.g. 'bluetooth', 'ollama')",
            required=True,
        ))

    async def execute(self, parameters: dict) -> ToolResult:
        service = parameters.get("service_name", "").strip()
        if not service:
            return ToolResult(
                success=False,
                error="Parameter 'service_name' is required.",
                metadata={"remedy_hint": "Provide a valid service name."},
            )

        # Implementation logic...
        return ToolResult(
            success=True,
            data={"service": service, "active": True},
        )
```

---

## 4. Execution Timeouts & Closed-Loop Recovery `[VERIFIED]`

Tools are subject to strict per-tier execution timeouts defined in [`zelyr/core/plugins.py`](../zelyr/core/plugins.py#L655-L666):
- **Tier 1 (IPC / OS):** 10.0s – 30.0s
- **Tier 2 (Browser / Web):** 10.0s
- **Tier 3 (Vision / SoM):** 35.0s

### Error Recovery Protocol:
When a tool fails or times out, the orchestrator parses the returned JSON. If `success` is `False`, the observation is automatically wrapped in a structured recovery envelope:
```
[Tool Error]: <error_message>. Recovery Hint: <remedy_hint>
```
The model is then prompted to execute the recommended recovery tool without outputting conversational apologies or non-actionable explanations.

---

## 5. Testing Tools Locally & Headlessly `[VERIFIED]`

When testing newly authored tools, adhere to the non-negotiable headless safeguards:
1. **Never spawn browser windows or desktop popups:** Mock `webbrowser.open` and subprocesses.
2. **Never connect to live external networks:** Mock `httpx.AsyncClient` or use `zelyr.security.network.validate_safe_url`.
3. **Execute unit tests using pytest:**
   ```bash
   uv run pytest -v tests/test_context_budget_fix.py
   ```
4. **Test Dynamic Loading via CLI:**
   ```bash
   uv run zelyr tool list
   uv run zelyr tool run inspect_git_status --args '{"repo_path": "."}'
   ```

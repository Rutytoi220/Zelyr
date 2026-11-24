# Zelyr Security Model & Execution Boundaries

This document defines the security architecture, execution model, trust assumptions, and isolation boundaries of Zelyr.

---

## 1. Threat Model & Trust Boundary

Zelyr is designed as a **single-user, local-first workstation orchestration system**.

- **Operator Authority:** The system assumes that the local developer running Zelyr is an authorized, trusted operator with standard user privileges on the workstation.
- **Single Host User UID:** All Zelyr processes—including the REPL, TUI, HTTP servers, and dynamic tool runners—execute under the host user's UID and GID.
- **Multi-Tenant Exclusion:** Zelyr is **not** designed for multi-tenant, untrusted remote code execution (RCE) environments where untrusted parties submit arbitrary prompts or tool code without supervision.

---

## 2. Dynamic Plugin Containment & Process Boundaries

### Honest Isolation Analysis
Built-in Ring-0 actuators run in-process (or via `asyncio.to_thread`) under the host user's UID/GID. Dynamic tools in `~/.config/zelyr/tools.d/` (`REQUIRED_RING >= 1`, validated by `zelyr/tools/validator.py` and executed via [`SandboxedPluginRunner`](../zelyr/core/plugin_sandbox.py)) run in a separate child process with resource and timeout limits under the same host user's UID/GID.

```
┌────────────────────────────────────────────────────────┐
│            Host Workstation (Local User UID)           │
│                                                        │
│  ┌───────────────────────┐   importlib.util (staging)  │
│  │ AST Structural Check  │ ──────────────────┐         │
│  │ (ToolValidator)       │                   ▼         │
│  └───────────────────────┘          ┌────────────────┐ │
│              │                      │ Staging Import │ │
│              ▼                      │ in /tmp/       │ │
│  ┌───────────────────────┐          └────────────────┘ │
│  │ Ring + Unified Policy │                   │         │
│  │ (No Ring 0 escalation)│                   ▼         │
│  └───────────────────────┘     ┌─────────────────────┐ │
│                                │ SandboxedPluginRunner│ │
│                                │ Child Process       │ │
│                                │ (RLIMIT_AS=512MB,   │ │
│                                │  RLIMIT_CORE=0)     │ │
│                                └─────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

### What the Staging & Execution Boundary DOES Enforce:
1. **AST Structural Verification:** `ToolValidator.validate_syntax_and_ast()` parses candidate code with `ast.parse()` to catch syntax errors, verify that `execute` is defined as an `async def` function, and ensure mandatory metadata (`TOOL_SCHEMA`, `REQUIRED_RING`, `TIER`, `TUI_HINT`) exists.
2. **OpenAI Function Spec Conformance:** Validates that `TOOL_SCHEMA` contains valid parameter types, property dictionaries, and string names.
3. **Privilege Escalation Gate:** Generated dynamic tools cannot claim `REQUIRED_RING = 0` (Ring 0 is strictly reserved for built-in, immutable system actuators). All dispatches also pass `authorize_and_validate_tool_call` and `ExecutionPolicyEngine`.
4. **Isolated Staging Dry-Run:** Injects candidate code into an ephemeral scratch file in `/tmp/zelyr_stage_<uuid>.py`, imports it dynamically via `importlib.util.spec_from_file_location`, tests `execute()` with default/test parameters under a 5.0-second timeout, and immediately cleans up the scratch file and `sys.modules` entry.
5. **Child-Process Crash & Memory Containment:** At runtime, dynamic plugins (`REQUIRED_RING >= 1`) run in a child process managed by `SandboxedPluginRunner` with `RLIMIT_AS` (512 MB), `RLIMIT_CORE = 0`, `/dev/null` stdin, and bounded timeouts (`SIGTERM` -> `SIGKILL`), plus atomic rollback if reloading fails.

### What `SandboxedPluginRunner` DOES NOT Provide:
> [!CAUTION]
> **No Kernel-Level Sandboxing:** AST checks, staging dry-runs, and `RLIMIT_AS`/`RLIMIT_CORE` child processes do **NOT** constitute an OS kernel sandbox (such as Linux user/mount/network namespaces, `seccomp-bpf`, `bubblewrap`, or `chroot`).
> - Child plugin processes inherit the host user's UID and filesystem/network permissions.
> - If executing untrusted code or tasks from unknown origins, isolate the workstation inside a dedicated VM or hardened container environment.

---

## 3. Evaluation Harness Path Traversal Containment

The benchmark evaluation harness (`zelyr.eval`) runs tasks against isolated ephemeral workspaces inside `/tmp/eval_workspace_<uuid>/`.

- **Path Resolution Boundary:** `EvalContext.resolve_path(rel_path)` resolves candidate file targets and enforces `path.is_relative_to(workspace_dir.resolve())`.
- **Traversal Prevention:** Any relative path containing traversal sequences (e.g., `../../etc/shadow` or directory escapes) raises a strict `ValueError("Path traversal detected")`.
- **Database Isolation:** Benchmark memory stores use ephemeral SQLite databases inside the temporary workspace, preventing mutation of the user's persistent `~/.config/zelyr/zelyr.db`.

---

## 4. Three-Tier Actuator Guardrails

Zelyr structures physical actuation into three deterministic tiers with explicit precedence rules to minimize destructive blind actions:

```
┌──────────────────────────────────────────────────────────────┐
│                    Actuator Precedence                       │
└──────────────────────────────────────────────────────────────┘
  │
  ├─► Tier 1: Deterministic System IPC (hyprctl, dbus, pactl)
  │   - Primary path for window management, media, notifications
  │   - Zero vision hallucination risk; structured JSON responses
  │
  ├─► Tier 2: Semantic Browser DOM Bridge (ws://127.0.0.1:41144)
  │   - Active when web browser window is focused
  │   - Dispatches DOM queries, semantic clicks, and text typing
  │   - Purges Tier 3 generic desktop vision to prevent UI clobbering
  │
  └─► Tier 3: Visual Grounding Fallback (Set-of-Mark / OCR / VLM)
      - Lowest precedence fallback for legacy desktop canvases & games
      - Strictly blocked in browser windows when DOM bridge is connected
```

---

## 5. Network Egress & Telemetry Invariant

- **Zero External Telemetry:** Zelyr transmits zero analytic telemetry, error reports, or prompt logs to external third-party cloud servers.
- **Local-First Default Inference:** Model generation and tool calls communicate strictly with local inference daemons over loopback (`http://127.0.0.1:11434` for Ollama).
- **Offline By Default:** Automated pytest suites (`uv run pytest -v`) and Level 1 benchmarks run 100% offline with mock stream providers (`tests/conftest.py` blocks outbound HTTP IO and redirects home/config dirs to temporary directories).

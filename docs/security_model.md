# Zelyr Security Model & Execution Boundaries

This document defines the security architecture, execution model, trust assumptions, and isolation boundaries of Zelyr.

---

## 1. Threat Model & Trust Boundary

Zelyr is designed as a **single-user, local-first workstation orchestration system**.

- **Operator Authority:** The system assumes that the local developer running Zelyr is an authorized, trusted operator with standard user privileges on the workstation.
- **Single Host User UID:** All Zelyr processes—including the REPL, TUI, HTTP servers, and dynamic tool runners—execute under the host user's UID and GID.
- **Multi-Tenant Exclusion:** Zelyr is **not** designed for multi-tenant, untrusted remote code execution (RCE) environments where untrusted parties submit arbitrary prompts or tool code without supervision.

---

## 2. In-Process Tool Execution & Sandbox Limits

### Honest Isolation Analysis
Dynamic tools crafted via `zelyr/tools/tool_crafter.py` and managed via `zelyr/tools/tool_manager.py` run **in-process** within the active Python interpreter.

```
┌────────────────────────────────────────────────────────┐
│               Host Workstation Process                 │
│              (Runs as Local User UID)                  │
│                                                        │
│  ┌───────────────────────┐   importlib.util            │
│  │ AST Structural Check  │ ──────────────────┐         │
│  │ (ToolValidator)       │                   ▼         │
│  └───────────────────────┘          ┌────────────────┐ │
│              │                      │ Staging Import │ │
│              ▼                      │ in /tmp/       │ │
│  ┌───────────────────────┐          └────────────────┘ │
│  │ Schema & Ring Conformance│                 │         │
│  │ (No Ring 0 escalation)│                   ▼         │
│  └───────────────────────┘          ┌────────────────┐ │
│                                     │ In-Process     │ │
│                                     │ async execute()│ │
│                                     └────────────────┘ │
└────────────────────────────────────────────────────────┘
```

### What the Staging Pipeline DOES Enforce:
1. **AST Structural Verification:** `ToolValidator.validate_syntax_and_ast()` parses candidate code with `ast.parse()` to catch syntax errors, verify that `execute` is defined as an `async def` function, and ensure mandatory metadata (`TOOL_SCHEMA`, `REQUIRED_RING`, `TIER`, `TUI_HINT`) exists.
2. **OpenAI Function Spec Conformance:** Validates that `TOOL_SCHEMA` contains valid parameter types, property dictionaries, and string names.
3. **Privilege Escalation Gate:** Generated dynamic tools cannot claim `REQUIRED_RING = 0` (Ring 0 is strictly reserved for built-in, immutable system actuators).
4. **Isolated Staging Dry-Run:** Injects candidate code into an ephemeral scratch file in `/tmp/zelyr_stage_<uuid>.py`, imports it dynamically via `importlib.util.spec_from_file_location`, tests `execute()` with default/test parameters under a 5.0-second timeout, and immediately cleans up the scratch file and `sys.modules` entry.
5. **Atomic Deployment & Rollback:** Installs new tools to `~/.config/zelyr/tools.d/{name}.py` via atomic file renaming, rolling back to previous versions if module reloading fails.

### What the Staging Pipeline DOES NOT Provide:
> [!CAUTION]
> **No Kernel-Level Sandboxing:** AST filtering, staging directories, and module unlinking do **NOT** constitute a kernel sandbox (such as Linux cgroups, user namespaces, seccomp-bpf, bubblewrap, or WASM isolation).
> - Dynamic tools run in the host process with standard user filesystem, process, and network capabilities.
> - An injected tool executing `os.system()` or `subprocess.Popen()` can read, modify, or delete any file accessible to the user running Zelyr.
> - If executing untrusted code or tasks from unknown origins, the entire workstation must be isolated inside a dedicated VM or container (e.g., Distrobox/Podman).

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
- **Offline By Default:** Automated test suites (`python3 -m unittest`) and Level 1 benchmarks run 100% offline with mock stream providers, ensuring tests never require network egress or running external services.

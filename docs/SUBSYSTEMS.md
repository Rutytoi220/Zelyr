# Zelyr Subsystem Catalog & Module Reference

This document provides a comprehensive, module-by-module reference of verified subsystems across **Zelyr** (v1.0.0).  
Each section specifies module responsibilities, public interfaces, state mutations, failure modes, and backing unit tests.

---

## 1. CLI & User Interfaces

### A. Universal CLI Entry Point ([`zelyr/cli/main.py`](../zelyr/cli/main.py))
- **Primary Responsibilities:**  
  Acts as the primary executable entry point (`zelyr`). Resolves command-line arguments and dispatches to appropriate UI modes, diagnostic utilities, or headless test runners.
- **Non-Responsibilities:**  
  Does not execute LLM inference directly; delegates to `repl.py`, `tui.py`, or `app.py`.
- **Verified Public Interfaces:**  
  `main() -> None`: Main console dispatcher.  
  `_build_parser() -> ArgumentParser`: Configures CLI arguments (`repl`, `chat`, `ui`, `server`, `run`, `send`, `status`, `test`, `daemon`, `serve`, `tool list`, `tool run`).
- **Dependencies:** `argparse`, `rich` (optional graceful degradation), `zelyr.config`.
- **State Mutations:** Triggers `zelyr.config.initialize_directories()`.
- **Failure Modes:** Falls back to standard `print()` if `rich` is unavailable.
- **Backing Tests:** [`tests/test_install_bootstrap.py:test_pyproject_entrypoints_and_callables`](../tests/test_install_bootstrap.py).

### B. Modern Streaming REPL ([`zelyr/cli/repl.py`](../zelyr/cli/repl.py))
- **Primary Responsibilities:**  
  Provides an inline, non-fullscreen streaming REPL with Tokyo Night aesthetics, prompt auto-suggestion, slash command autocompletion, and reasoning `<think>` token toggles.
- **Verified Public Interfaces:**  
  `run_repl() -> None`: Starts interactive prompt session.  
  `COMMANDS: Dict[str, str]`: Slash command registry (`/help`, `/model`, `/session`, `/thought`, `/tools`, `/memory`, `/doctor`, `/undo`, `/exit`).
- **Dependencies:** `prompt_toolkit`, `rich`, `zelyr.agents.native_orchestrator.NativeOrchestrator`.
- **Backing Tests:** Interactive verification via pseudo-terminal (PTY) harnesses.

### C. Desktop Graphical Client ([`zelyr/gui/app.py`](../zelyr/gui/app.py))
- **Primary Responsibilities:**  
  Thin PySide6 client running the desktop interface with dynamic QSS themes (`ThemeManager`) and non-blocking asynchronous event bridging (`ZelyrGuiBridge`).
- **Verified Public Interfaces:**  
  `main() -> None`: Initializes PySide6 application loop.
- **Dependencies:** `PySide6.QtWidgets`, `qasync`.
- **Failure Modes:** Exits with code 1 if PySide6 or X11/Wayland display server is inaccessible.
- **Backing Tests:** Headless Qt testing via `pytest-qt` and `QT_QPA_PLATFORM=offscreen`.

---

## 2. Core Orchestration & Routing

### A. Native Orchestrator ([`zelyr/agents/native_orchestrator.py`](../zelyr/agents/native_orchestrator.py))
- **Primary Responsibilities:**  
  Runs the streaming ReAct loop. Handles sliding-window context budgeting, window context introspection (`hyprctl`), tier-aware tool filtering, tolerant JSON/XML tool call extraction, and closed-loop error recovery.
- **Verified Public Interfaces:**  
  `NativeOrchestrator(max_context_tokens, max_steps, stream_provider, tool_schemas, tool_executor)`  
  `generate_stream(payload, depth, ...) -> AsyncGenerator[dict, None]`: Streams model deltas and yields tool observations.  
  `route_tier(task_description, active_window, bridge_connected) -> str`: Resolves automation tier.  
  `prune_messages(messages, max_context_tokens, reserve_tokens) -> list[dict]`: Applies sliding window context pruning.  
  `extract_first_tool_call(text, known_tools) -> Optional[dict]`: Extracts first valid tool call from prose or raw JSON.
- **Dependencies:** `httpx`, `zelyr.config`, `zelyr.core.plugins`, `zelyr.agents.actuator_arbiter`.
- **State Mutations:** Appends assistant tool calls and tool observation messages to the session history in `zelyr.db.memory`.
- **Failure Modes:** If Ollama is unreachable, yields structured `[Connection Error]` chunk without raising uncaught exceptions.
- **Backing Tests:** [`tests/test_context_budget_fix.py`](../tests/test_context_budget_fix.py), [`tests/test_multi_turn_payload.py`](../tests/test_multi_turn_payload.py).

### B. Actuator Arbiter ([`zelyr/agents/actuator_arbiter.py`](../zelyr/agents/actuator_arbiter.py))
- **Primary Responsibilities:**  
  Arbitrates between Tier 1 (IPC/OS), Tier 2 (Browser DOM), and Tier 3 (Vision Grounding) based on user query intent, active window class/title, and browser bridge state.
- **Verified Public Interfaces:**  
  `ActuatorArbiter.arbitrate(task, active_window, bridge_connected) -> str`: Returns `'tier1_ipc'`, `'tier2_browser'`, or `'tier3_vision'`.  
  `is_browser_window(active_window) -> bool`: Checks whether active window is a known browser.
- **Backing Tests:** [`tests/test_context_budget_fix.py`](../tests/test_context_budget_fix.py).

### C. Fastest Path Tool Router ([`zelyr/core/fast_path_router.py`](../zelyr/core/fast_path_router.py))
- **Primary Responsibilities:**  
  Intercepts standard deterministic actions (URLs, known binary launches, systemctl units) to execute them directly via CLI, bypassing LLM inference entirely.
- **Verified Public Interfaces:**  
  `FastestPathRouter.can_route_to_cli(action, params) -> Optional[FastPathPlan]`: Returns execution plan or `None` if GUI automation is required.
- **Backing Tests:** [`tests/test_fast_path_router.py`](../tests/test_fast_path_router.py).

---

## 3. Tool Plugin System & Dynamic Provisioning

### A. Plugin Core ([`zelyr/core/plugins.py`](../zelyr/core/plugins.py))
- **Primary Responsibilities:**  
  Provisions built-in tools into `~/.config/zelyr/tools.d/` on initialization, dynamically loads external plugins, registers capability metadata, and filters schemas by tier.
- **Verified Public Interfaces:**  
  `load_plugins() -> None`: Discovers and reloads all `.py` plugin files in `TOOLS_DIR`.  
  `get_tool_schemas(ring, tier, is_browser) -> List[Dict[str, Any]]`: Returns filtered OpenAPI schemas.  
  `filter_tool_schemas_by_tier(schemas, tier, is_browser) -> List[Dict[str, Any]]`: Prunes tools outside the active tier.  
  `get_tool_capability(name) -> ToolCapability`: Returns capability metadata (tier, token cost, requirements).  
  `get_tier_timeout(func_name) -> float`: Returns timeout seconds (Tier 1: 10s-30s, Tier 2: 10s, Tier 3: 35s).
- **Provisioned Built-In Tools:**
  1. `remember_fact`: Stores permanent facts into SQLite semantic memory.
  2. `manage_system_process`: Lists, inspects, and terminates Linux processes.
  3. `manage_system_clipboard`: Reads and writes Wayland/X11 clipboard via `wl-copy`/`wl-paste`.
  4. `query_system_journal`: Queries `journalctl` with unit and priority filtering.
  5. `manage_media_playback`: Controls desktop audio and MPRIS media players via `playerctl`.
  6. `manage_workspace_file`: Reads, creates, and applies bounded replacements to local files.
  7. `send_desktop_notification`: Dispatches native desktop notifications via `notify-send`.
  8. `inspect_network`: Checks network interfaces, default gateways, and latency.
  9. `execute_command`: Executes shell commands via bash.
  10. `manage_desktop_window`: Controls workspace, focus, and tiling via `hyprctl`.
- **Backing Tests:** [`tests/test_context_budget_fix.py`](../tests/test_context_budget_fix.py), [`tests/test_install_bootstrap.py`](../tests/test_install_bootstrap.py).

---

## 4. Security & Isolation Boundaries

### A. SSRF Network Security Guard ([`zelyr/security/network.py`](../zelyr/security/network.py))
- **Primary Responsibilities:**  
  Validates all outbound URLs before network connection to prevent Server-Side Request Forgery (SSRF). Enforces per-hop IP validation across HTTP redirects.
- **Verified Public Interfaces:**  
  `validate_safe_url(url: str) -> str`: Validates scheme, resolves hostname to IP addresses, and checks prohibited ranges.  
  `fetch_with_ssrf_guard(url, client, max_redirects) -> httpx.Response`: Fetches URL with per-hop redirect validation.
- **Blocked Address Ranges:**
  - Non-HTTP/HTTPS schemes (`file://`, `ftp://`, `gopher://`).
  - Loopback IPs (`127.0.0.0/8`, `::1`).
  - Private subnets (RFC 1918: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`).
  - Cloud metadata endpoints (`169.254.169.254`, `fe80::/10`).
  - Carrier-Grade NAT & Tailscale private mesh IPs (RFC 6598: `100.64.0.0/10`).
  - Current network / local host (RFC 1122: `0.0.0.0/8`).
  - IPv6 Documentation (RFC 3849: `2001:db8::/32`).
- **Backing Tests:** [`tests/test_web_search.py:TestSSRFSecurity`](../tests/test_web_search.py).

### B. Shell Authorization Confirmation ([`zelyr/tools/__init__.py`](../zelyr/tools/__init__.py))
- **Primary Responsibilities:**  
  Guards arbitrary shell command execution behind an explicit human confirmation prompt in interactive TTY sessions. Fails closed in non-interactive environments.
- **Verified Public Interfaces:**  
  `_request_shell_authorization(tool_id: str, command: str) -> bool`: Prompts via `rich.prompt.Confirm.ask`.
- **Bypass Conditions:**  
  Returns `True` without prompting only if `ZELYR_AUTOPILOT="1"` or `ZELYR_TESTING="1"` is set.
- **Backing Tests:** [`tests/test_shell_authorization.py`](../tests/test_shell_authorization.py).

---

## 5. Storage & Swarm Memory

### A. SQLite Persistent Memory ([`zelyr/db/memory.py`](../zelyr/db/memory.py))
- **Primary Responsibilities:**  
  Stores conversational session history, messages, and semantic memories in SQLite with Write-Ahead Logging (`WAL`).
- **Verified Public Interfaces:**  
  `create_session(session_id, db_path) -> None`: Initializes a new conversation session.  
  `add_message(session_id, role, content, db_path) -> None`: Persists a message turn.  
  `get_session_messages(session_id, db_path) -> List[Dict[str, Any]]`: Retrieves session history.  
  `remember_fact(fact, category, db_path) -> None`: Inserts facts with full-text search indexing (`FTS5`).

### B. Swarm Blackboard Store ([`zelyr/memory/blackboard.py`](../zelyr/memory/blackboard.py))
- **Primary Responsibilities:**  
  Maintains in-memory key-value state for multi-agent swarm collaboration with strict session and agent isolation.
- **Verified Public Interfaces:**  
  `write(session_id, agent_id, key, value) -> None`: Writes an artifact to the agent's private namespace.  
  `read(session_id, agent_id, key, default) -> Any`: Reads from the agent's private namespace.  
  `list_keys(session_id, agent_id) -> List[str]`: Lists an agent's registered keys.  
  `purge_session(session_id) -> None`: Purges all keys and namespaces associated with a session.
- **Backing Tests:** [`tests/test_swarm_blackboard.py`](../tests/test_swarm_blackboard.py).

---

## 6. System Configuration & Directory Bootstrap

### A. Configuration Module ([`zelyr/config.py`](../zelyr/config.py))
- **Primary Responsibilities:**  
  Resolves environment overrides, initializes required directory structures on first launch, and provides clamped context lengths.
- **Verified Public Interfaces:**  
  `get_config_dir() -> Path`: Resolves `ZELYR_CONFIG_DIR` or defaults to `~/.config/zelyr`.  
  `initialize_directories(config_dir: Optional[Path] = None) -> None`: Creates `tools.d/`, `plugins/`, `sessions/`, `logs/`, `models/`.  
  `validate_and_clamp_num_ctx(val: Any) -> int`: Validates context integers within `[1024, 65536]`, defaulting to `8192`.  
  `get_effective_num_ctx(caller_override: Optional[Any] = None) -> int`: Resolves context with 4-level precedence.
- **Backing Tests:** [`tests/test_install_bootstrap.py`](../tests/test_install_bootstrap.py), [`tests/test_multi_turn_payload.py`](../tests/test_multi_turn_payload.py).

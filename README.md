# Zelyr

[![Platform: Linux/Wayland](https://img.shields.io/badge/platform-Linux%20%2F%20Wayland-blue.svg)](https://github.com/Rutytoi220/Zelyr)
[![Python: 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Local Inference: Ollama](https://img.shields.io/badge/inference-Ollama%20127.0.0.1%3A11434-orange.svg)](https://ollama.com/)

**Zelyr** is a local-first autonomous AI orchestration engine built for Linux (specifically Wayland compositors such as Hyprland). It connects local reasoning models to physical desktop workflows through structured system IPC, semantic web DOM automation, persistent memory, and hot-reloadable dynamic tool crafting.

Designed to operate on consumer hardware (e.g., RTX 4060 8GB VRAM) with zero external cloud dependencies, Zelyr executes inference entirely against local endpoints (`http://127.0.0.1:11434`) with zero cloud telemetry.

---

## 1. Core Architecture & Subsystems

```mermaid
flowchart TD
    User([User Prompt / Task]) --> Orchestrator["NativeOrchestrator (ReAct Loop)"]
    Orchestrator --> ContextBudget["Unified Context Budget (8K num_ctx)"]
    
    subgraph Actuator ["Three-Tier Actuator Hierarchy"]
        direction TB
        T1["Tier 1: System IPC\n(hyprctl / dbus / pactl)"]
        T2["Tier 2: Semantic Browser DOM\n(WebExtension Bridge ws://127.0.0.1:41144)"]
        T3["Tier 3: Vision Fallback\n(Set-of-Mark / OCR / VLM)"]
        T1 -->|Desktop App Focus| T1
        T2 -->|Browser Active: Purges Tier 3| T2
        T3 -->|Legacy UI / Canvas Games Only| T3
    end
    
    subgraph Storage ["Persistent Subsystems"]
        Memory["Persistent Memory Store\n(SQLite WAL + FTS5 Index)"]
        ToolManager["Tool Crafter & Manager\n(~/.config/zelyr/tools.d/)"]
    end
    
    Orchestrator --> Actuator
    Orchestrator <--> Memory
    Orchestrator <--> ToolManager
```

### A. Three-Tier Actuator Hierarchy & Negative Gating
Physical workstation control is partitioned into three deterministic tiers with explicit precedence rules to avoid destructive or hallucinated UI actions:

1. **Tier 1: Deterministic System IPC (Highest Precedence)**
   - Interacts directly with Linux subsystems via native IPC (`hyprctl`, `dbus`, `pactl`, `systemd`).
   - Tools: `manage_desktop_window`, `manage_system_process`, `manage_workspace_file`, `query_system_journal`, `manage_media_playback`, `send_desktop_notification`.
   - Returns structured JSON; zero visual hallucination risk.

2. **Tier 2: Semantic Browser DOM & Web Search**
   - Interacts with active web pages through an authenticated local WebSocket server (`ws://127.0.0.1:41144`) connected to the Zelyr WebExtension.
   - Native Web Tools: `web_search` (query DuckDuckGo / SearXNG with context budgeting) and `fetch_web_page` (HTML text extraction with SSRF security and prompt injection isolation boundaries).
   - Tools: `interact_with_browser`, `web_search`, `fetch_web_page`.
   - **Negative Gating & Tier Pruning:** When a browser window is focused and the DOM bridge is active, `NativeOrchestrator` strictly purges Tier 3 generic desktop vision tools (`interact_with_ui`, `capture_desktop_vision`) from the LLM tool manifest. This prevents models from guessing pixel coordinates on interactive web pages.

3. **Tier 3: Visual Grounding Fallback (Lowest Precedence)**
   - Set-of-Mark (SoM) bounding box detection, OCR, and vision-language model (VLM) grounding.
   - Reserved strictly as a fallback for non-DOM applications (legacy X11/Qt widgets, canvas games, remote desktops) where no IPC or DOM introspection exists.

---

### B. Persistent Memory Store (SQLite WAL + FTS5)
Persistent memory is implemented via [`zelyr/db/memory.py`](zelyr/db/memory.py):
- **Concurrency & Resilience:** SQLite configured with Write-Ahead Logging (`PRAGMA journal_mode=WAL`), high busy timeouts (`PRAGMA busy_timeout=5000`), and foreign key enforcement.
- **Full-Text Retrieval:** Virtual FTS5 tables (`memories_fts`) with triggers automatically index stored facts.
- **Categorization & Versioning:** Enforces strict categories (`preference`, `environment`, `homelab`, `workflow`, `capability`). Superseded facts track historical lineage without data loss.
- **Session Fact Promotion:** Automatically hydrates relevant memories into initial conversational context and records structured facts across reboots.

---

### C. Autonomous Dynamic Tool Crafting & Lifecycle Management
Zelyr synthesizes, validates, and installs new tools at runtime into `~/.config/zelyr/tools.d/` via [`zelyr/tools/tool_crafter.py`](zelyr/tools/tool_crafter.py) and [`zelyr/tools/tool_manager.py`](zelyr/tools/tool_manager.py):
- **AST Contract Inspection:** Statically parses candidate code with Python's `ast` module to confirm valid syntax, required `async def execute` definitions, and standard schema exports (`TOOL_SCHEMA`, `REQUIRED_RING`, `TIER`, `TUI_HINT`).
- **Privilege Escalation Protection:** Generated dynamic tools cannot claim `REQUIRED_RING = 0` (Ring 0 is immutable and reserved for built-in actuators).
- **Isolated Staging Dry-Run:** Stages the code into `/tmp/zelyr_stage_<uuid>.py`, dynamically imports it via `importlib.util.spec_from_file_location`, tests execution under a 5.0s timeout, and cleans up `sys.modules` and scratch files.
- **Atomic Rollback:** If plugin reloading fails, the file is unlinked, the runtime schema is unregistered, and persistent memory is restored to the previous working version.

---

### D. Benchmark & Evaluation Harness (`zelyr.eval`)
Zelyr includes a deterministic and live benchmark evaluation suite via [`zelyr/eval/`](zelyr/eval/):
- **Level 1 (Deterministic Offline):** 6 self-contained tasks evaluating tool dispatch, recovery envelopes, step limits, false success rejection, memory roundtrips, and timeouts in under 3 seconds with zero GPU or network dependencies.
- **Level 2 (Live Model Evaluation):** Exercises real multi-turn ReAct loops against local models (`qwen3:8b`, `qwen3:1.7b`) with independent disk and state validators.
- **Level 3 (Complex Workflows):** Tests multi-step triage, tool self-repair, and containment.
- **Anti-Hallucination Outcome Verification:** Rejects verbal conversational claims if real observable side-effects (e.g. disk artifacts) were not produced, tagging them as `FALSE_SUCCESS`.
- **Measured Token Telemetry:** Distinguishes measured prompt and evaluation tokens reported by the Ollama stream from character-based estimations.

---

## 2. Benchmark Baseline Highlights (RTX 4060 8GB VRAM)

Empirical results from [`docs/benchmarks/live_baseline_report.md`](docs/benchmarks/live_baseline_report.md) executed on consumer hardware (Linux Fedora Atomic, NVIDIA RTX 4060 Laptop 8GB VRAM, Ollama v0.40.1):

| Model | Task ID | Steps | Measured Tokens | Duration | Status | Outcome / Behavioral Verification |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **qwen3:8b** | `task_live_workspace_diagnostic` | 3 | 12,159 | 60.03s | **PASS** | Complete multi-turn ReAct (`read_file` -> `write_file` -> verified `diagnosis.json`). |
| **qwen3:8b** | `task_live_metric_formatting` | 3 | 12,444 | 58.02s | **PASS** | Multi-turn ReAct (`read_file` -> `write_file` -> verified `summary.json`). |
| **qwen3:8b** | `task_synthetic_service_triage` | 4 | 15,333 | 52.56s | **FALSE_SUCCESS** | Completed 4 turns, 3 tool calls; verbal claim without exact state trigger. |
| **qwen3:8b** | `task_multisession_memory_persistence` | 3 | 11,748 | 47.76s | **FAIL** | Completed 3 turns; verbal completion without storing key to persistent memory. |
| **qwen3:8b** | `task_unrecoverable_actuator_containment` | 3 | 8,188 | 60.04s | **TIMEOUT** | Deep reasoning turn reached 60.0s boundary cleanly. |
| **qwen3:1.7b** | `task_live_workspace_diagnostic` | 3 | 11,133 | 13.81s | **PASS** | Fast multi-turn ReAct diagnostic in 13.8s. |
| **qwen3:1.7b** | `task_live_metric_formatting` | 3 | 11,360 | 10.44s | **FAIL** | Trailing explanatory prose outside JSON boundaries. |

> **Context Sizing Invariant:** Rightsizing the default context to `num_ctx: 8192` eliminates CUDA Out-Of-Memory (OOM) crashes on 8GB GPUs, keeping total VRAM utilization during inference under ~6.5 GB.

---

## 3. Quickstart

### Prerequisites
- **Operating System:** Linux (Hyprland / Wayland recommended; X11 supported for non-compositor tools).
- **Python:** Python 3.10 or 3.11.
- **Package Manager:** [`uv`](https://docs.astral.sh/uv/) (recommended) or standard `pip`.
- **Inference Engine:** [Ollama](https://ollama.com/) running on `http://127.0.0.1:11434`.

### Installation & Execution

```bash
# 1. Clone repository
git clone https://github.com/Rutytoi220/Zelyr.git
cd Zelyr

# 2. Sync virtual environment and dependencies (Core runtime)
uv sync

# Optional extras:
# Install with machine learning / local VLM support:
uv sync --extra ml
# Install with browser automation support:
uv sync --extra browser
# Install all optional features (audio, automation, experimental, ml, browser):
uv sync --extra all

# Or via standard pip:
pip install zelyr-core               # Core runtime
pip install 'zelyr-core[ml]'         # Core + Local VLM / PyTorch
pip install 'zelyr-core[browser]'    # Core + Playwright Web Engine
pip install 'zelyr-core[all]'        # Full feature battery

# 3. Run the automated test suite
uv run pytest -v

# 4. Launch Zelyr interactive streaming CLI
uv run zelyr
```

---

## 4. Technical Documentation Hub

For developers, architects, and AI coding agents navigating the codebase, refer to the source-verified guides in [`docs/`](docs/):

| Guide | Description | Target Audience |
| :--- | :--- | :--- |
| **[`docs/INDEX.md`](docs/INDEX.md)** | **Central AI & Developer Navigation Map** | Coding agents, contributors |
| **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** | Runtime architecture, request lifecycle, context budgeting | Architects, developers |
| **[`docs/SUBSYSTEMS.md`](docs/SUBSYSTEMS.md)** | Subsystem catalog, module responsibilities, public APIs | Developers, maintainers |
| **[`docs/TOOLS.md`](docs/TOOLS.md)** | Tool creation guide, dynamic plugin format, capability tiers | Tool authors, extenders |
| **[`docs/SECURITY.md`](docs/SECURITY.md)** | SSRF protection, prompt injection containment, shell auth | Security auditors, engineers |
| **[`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md)** | Developer workflows, headless testing, packaging | All contributors |
| **[`AGENTS.md`](AGENTS.md)** | Behavioral & empirical verification standard for AI agents | Autonomous coding agents |

---

## 5. CLI Interface & Commands

Zelyr provides multiple interfaces accessible via `zelyr` (registered entrypoint) or `uv run zelyr`:

```bash
# Modern inline streaming CLI (Tokyo Night theme, default)
zelyr repl

# Full-screen Opencode-tier TUI
zelyr chat

# PySide6 Desktop GUI
zelyr ui

# Start background FastAPI compute node & OpenAI-compatible proxy
zelyr server

# Run autonomous agent with natural-language prompt
zelyr run "Inspect open windows and list system memory"

# List all registered and dynamic tools
zelyr tool list

# Execute a tool directly by ID
zelyr tool run query_system_journal --args '{"lines": 10}'

# Show system and daemon telemetry
zelyr status

# Run automated headless test suite
zelyr test

# Run deterministic benchmark suite
python -m zelyr.eval --level 1

# Run live benchmark evaluation against local Ollama model
python -m zelyr.eval --level 2 --live -m qwen3:8b --json benchmark_report.json
```

---

## 6. Security & Isolation Model

Zelyr operates under a transparent, single-user workstation security model:
- **SSRF Network Protection:** Outbound network requests pass through [`zelyr.security.network.validate_safe_url`](zelyr/security/network.py), blocking loopback, RFC 1918 private subnets, cloud metadata (169.254.169.254), Carrier-Grade NAT / Tailscale mesh (`100.64.0.0/10`), and re-validating redirect hops.
- **Prompt Injection Isolation:** Web content and search results are enclosed in `<external_web_content>` XML boundaries with tag escaping.
- **Fail-Closed Shell Confirmation:** Arbitrary shell commands require explicit interactive user approval via [`zelyr/tools/__init__.py`](zelyr/tools/__init__.py) (`_request_shell_authorization`) and fail closed in non-interactive sessions.
- **In-Process Execution Reality:** Dynamic tools execute in-process inside the Python interpreter under the host user's UID. AST validation and `/tmp` staging verify syntax and prevent accidental errors, but do **not** constitute an OS-level kernel sandbox.
- **Zero External Telemetry:** All inference traffic remains strictly local on loopback (`http://127.0.0.1:11434`). Zero analytics, error reports, or prompts are sent to external servers.

For full technical specifications, read [`docs/SECURITY.md`](docs/SECURITY.md) and [`docs/security_model.md`](docs/security_model.md).

---

## 7. License

Distributed under the MIT License. See `LICENSE` for more information.

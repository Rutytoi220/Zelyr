# Zelyr Documentation Index & AI Navigation Map

Welcome to the technical documentation map for **Zelyr** (v1.0.0).  
This index serves as the primary navigation hub for software engineers and local coding agents (including smaller models running via OpenCode and Ollama) navigating the repository.

Every statement, code path, and interface in these documents is grounded in verified source code, active configurations, and observed test executions.

---

## 1. Quick Navigation by Objective

| Developer Objective | Recommended Reading | Primary Source Files |
| :--- | :--- | :--- |
| **Trace a User Request** | [`ARCHITECTURE.md: Request Lifecycle`](ARCHITECTURE.md#2-request-lifecycle-execution-flow-verified) | [`zelyr/cli/main.py`](../zelyr/cli/main.py), [`zelyr/agents/native_orchestrator.py`](../zelyr/agents/native_orchestrator.py) |
| **Add or Modify a Tool** | [`TOOLS.md: Tool Creation & Extension`](TOOLS.md) | [`zelyr/core/plugins.py`](../zelyr/core/plugins.py), [`zelyr/tools/base.py`](../zelyr/tools/base.py) |
| **Understand Subsystems** | [`SUBSYSTEMS.md: Subsystem Catalog`](SUBSYSTEMS.md) | [`zelyr/`](../zelyr/) |
| **Inspect Security & SSRF** | [`SECURITY.md: Security Architecture`](SECURITY.md) | [`zelyr/security/network.py`](../zelyr/security/network.py), [`zelyr/tools/__init__.py`](../zelyr/tools/__init__.py) |
| **Manage Context & Tokens** | [`ARCHITECTURE.md: Context Budgeting`](ARCHITECTURE.md#4-context-budgeting-sliding-window-algorithm-verified) | [`zelyr/config.py`](../zelyr/config.py), [`zelyr/agents/native_orchestrator.py`](../zelyr/agents/native_orchestrator.py) |
| **Run Tests & Verify Headless** | [`DEVELOPER_GUIDE.md: Testing`](DEVELOPER_GUIDE.md#2-testing-verification-standards-verified) | [`tests/`](../tests/), [`tests/conftest.py`](../tests/conftest.py), [`pyproject.toml`](../pyproject.toml) |
| **Build Wheels & Bootstrap** | [`DEVELOPER_GUIDE.md: Release & Packaging`](DEVELOPER_GUIDE.md#4-packaging-distribution-bootstrapping-verified) | [`pyproject.toml`](../pyproject.toml), [`tests/test_install_bootstrap.py`](../tests/test_install_bootstrap.py) |
| **Agent Operational Rules** | [`../AGENTS.md`](../AGENTS.md) | [`AGENTS.md`](../AGENTS.md) |

---

## 2. Repository Topology & "Where to Look First"

When tasked with investigating or modifying Zelyr, start at these verified entry points:

```
ChienGPT/
├── AGENTS.md                  # Behavioral and empirical verification protocol for coding agents
├── README.md                  # Project overview, installation, and user CLI guide
├── pyproject.toml             # Python package manifest, dependencies, entry points, pytest config
├── docs/                      # Central technical documentation repository
│   ├── INDEX.md               # [This file] The central AI and developer navigation map
│   ├── ARCHITECTURE.md        # Runtime architecture, request lifecycle, context budgeting
│   ├── SUBSYSTEMS.md          # Module-by-module reference of verified subsystems
│   ├── TOOLS.md               # Tool creation guide, dynamic plugin format, capability tiers
│   ├── SECURITY.md            # SSRF protection, prompt injection boundaries, shell auth
│   └── DEVELOPER_GUIDE.md     # Development workflows, test suites, headless requirements
├── zelyr/                     # Core application source code
│   ├── agents/                # Orchestration engines and actuator arbitration
│   │   ├── native_orchestrator.py  # [VERIFIED] Active ReAct orchestrator used by CLI REPL/TUI
│   │   ├── actuator_arbiter.py     # [VERIFIED] Dynamic 3-tier capability router (IPC/DOM/Vision)
│   │   └── orchestrator_agent.py   # [PARTIALLY VERIFIED] Heavy state-machine agent
│   ├── cli/                   # Console interfaces and user entry points
│   │   ├── main.py            # [VERIFIED] Console script entry point (`zelyr`)
│   │   ├── repl.py            # [VERIFIED] Modern inline streaming CLI (Tokyo Night theme)
│   │   └── tui.py             # [VERIFIED] Fullscreen OpenCode-tier TUI
│   ├── config.py              # [VERIFIED] Configuration, path resolution, context clamping
│   ├── core/                  # Kernel components, fast-path routing, and plugin loader
│   │   ├── fast_path_router.py# [VERIFIED] Intercepts URL/app launches to bypass LLM
│   │   ├── plugins.py         # [VERIFIED] Dynamic plugin loader (~/.config/zelyr/tools.d/)
│   │   ├── events.py          # [VERIFIED] In-process synchronous/asynchronous EventBus
│   │   └── transaction.py     # [VERIFIED] Workspace transaction and staging rollback
│   ├── db/                    # Persistent storage engines
│   │   └── memory.py          # [VERIFIED] SQLite WAL session history and semantic facts
│   ├── gui/                   # Desktop graphical user interface
│   │   ├── app.py             # [VERIFIED] PySide6 desktop application entry point
│   │   └── bridge.py          # [VERIFIED] Asynchronous GUI bridge connecting Qt to IPC
│   ├── memory/                # Context memory hydration and blackboard
│   │   ├── hydration.py       # [VERIFIED] Injects top-k memories into system prompt
│   │   └── blackboard.py      # [VERIFIED] Multi-agent shared blackboard with namespace isolation
│   ├── security/              # Security and network validation
│   │   └── network.py         # [VERIFIED] SSRF guard, IP subnet validation, safe HTTP client
│   ├── server/                # Headless background service
│   │   └── daemon.py          # [VERIFIED] ZelyrDaemonServer background WebSocket daemon (`zelyrd`)
│   └── tools/                 # Tool implementations and base definitions
│       ├── base.py            # [VERIFIED] BaseTool, ToolParameter, ToolResult definitions
│       ├── core.py            # [VERIFIED] ToolCapability metadata definitions
│       ├── os_system.py       # [VERIFIED] Linux process, clipboard, and journal tools
│       ├── os_desktop.py      # [VERIFIED] Media control and desktop notification tools
│       ├── web_search.py      # [VERIFIED] Native web search and extracted text fetching
│       └── workspace_file.py  # [VERIFIED] Workspace file reading and bounded patching
└── tests/                     # Automated test suites (Strictly offline & headless)
    ├── conftest.py            # [VERIFIED] Headless fixtures, mock home dir, network blocking
    ├── test_context_budget_fix.py   # [VERIFIED] 8K context token limits & tier filtering
    ├── test_fast_path_router.py     # [VERIFIED] URL and app launching fast-path routing
    ├── test_install_bootstrap.py    # [VERIFIED] Portable paths, packaging, directory bootstrap
    ├── test_multi_turn_payload.py   # [VERIFIED] Context precedence and Turn 2 protocol compliance
    ├── test_shell_authorization.py  # [VERIFIED] Shell execution confirmation and fail-closed security
    ├── test_swarm_blackboard.py     # [VERIFIED] Multi-agent blackboard namespace isolation
    └── test_web_search.py           # [VERIFIED] Web search, extraction, SSRF, prompt injection tests
```

---

## 3. Empirical Confidence Rating Standards

All technical statements across these documents adhere to explicit confidence classifications:

- `[VERIFIED]`: Directly proven by source code inspection, active configuration, or automated test execution in the active environment.
- `[PARTIALLY VERIFIED]`: Supported by source code structure, but requires live hardware, active desktop sessions, or external daemons not currently running.
- `[INFERRED]`: Plausible architectural intent deduced from naming, design patterns, or partial references.
- `[UNKNOWN]`: Unverified or conflicting information in the current repository.

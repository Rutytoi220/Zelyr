# Zelyr Architecture & Execution Flow Specification

This document specifies the runtime architecture, request lifecycle, context budgeting, and tool routing mechanisms in **Zelyr** (v1.0.0).  
All architecture claims are grounded in verified source code and reproducible automated tests.

---

## 1. System Topology Overview

Zelyr is a local-first AI operating system engineered for Linux (specifically Wayland compositors such as Hyprland). It decouples user interfaces, background daemons, local LLM inference engines, and OS actuators:

```mermaid
flowchart TD
    subgraph UI ["User Interface Layer"]
        CLI["zelyr repl / chat (prompt_toolkit / rich)"]
        GUI["zelyr ui (PySide6 Desktop Client)"]
    end

    subgraph FastPath ["Bypass Layer"]
        Router["FastestPathRouter (Intercepts CLI/URL commands)"]
    end

    subgraph Orchestrator ["Orchestration Layer"]
        NativeOrch["NativeOrchestrator (Streaming ReAct Loop)"]
        Arbiter["ActuatorArbiter (Tier 1 / Tier 2 / Tier 3)"]
        ContextMgr["Sliding Window Context (Clamped 1024-65536, 8K Default)"]
    end

    subgraph Inference ["Local Inference Engine"]
        Ollama["Ollama API (http://127.0.0.1:11434)"]
    end

    subgraph Actuators ["Three-Tier Actuator Hierarchy"]
        T1["Tier 1: Linux IPC (hyprctl, dbus, systemctl)"]
        T2["Tier 2: Browser & Web (WebExtension ws://127.0.0.1:41144 + WebSearch)"]
        T3["Tier 3: Vision Fallback (Set-of-Mark / OCR / VLM)"]
    end

    subgraph Storage ["Persistent State & Memory"]
        SQLite["SQLite WAL Database (Sessions & Semantic Facts)"]
        Blackboard["Swarm BlackboardStore (Namespace-Isolated State)"]
        ToolsDir["Dynamic Plugin Directory (~/.config/zelyr/tools.d/)"]
    end

    UI --> Router
    Router -->|Non-AI Native Command| T1
    Router -->|Complex Reasoning Required| NativeOrch
    NativeOrch <--> Arbiter
    NativeOrch <--> ContextMgr
    NativeOrch <--> Ollama
    NativeOrch --> Actuators
    NativeOrch <--> Storage
```

---

## 2. Request Lifecycle & Execution Flow `[VERIFIED]`

When a user submits a prompt via [`zelyr.cli.repl`](../zelyr/cli/repl.py), [`zelyr.cli.tui`](../zelyr/cli/tui.py), or [`zelyr.gui.app`](../zelyr/gui/app.py), the execution follows a deterministic, 10-step lifecycle:

```mermaid
sequenceDiagram
    autonumber
    participant User as User / Shell
    participant Router as FastestPathRouter
    participant Orch as NativeOrchestrator
    participant Arbiter as ActuatorArbiter
    participant Ollama as Ollama (Local LLM)
    participant Tool as Tool Executor
    participant OS as Linux OS / Browser

    User->>Router: Submit Prompt ("open https://example.com")
    alt Can Route to Native CLI?
        Router->>OS: Execute xdg-open https://example.com
        Router-->>User: Instant Completion (Zero LLM Tokens)
    else Requires Autonomous Reasoning
        Router->>Orch: Forward to NativeOrchestrator.generate_stream()
        Orch->>OS: Query Active Window (hyprctl activewindow -j)
        OS-->>Orch: Window Title & Class Metadata
        Orch->>Arbiter: Arbitrate Automation Tier (Query + Window)
        Arbiter-->>Orch: Selected Tier (tier1_ipc, tier2_browser, tier3_vision)
        Orch->>Orch: Filter Tool Schemas & Prune Inactive Tiers
        Orch->>Orch: Apply Sliding Window Context Budget (Reserve 1000 tokens)
        Orch->>Ollama: POST /v1/chat/completions (Stream)
        Ollama-->>Orch: Streamed Chunks + Tool Call
        Orch->>Orch: Tolerant Extraction (extract_first_tool_call)
        Orch->>Tool: execute_tool(func_name, **func_args)
        Tool->>OS: Perform System Action / Web Fetch
        OS-->>Tool: Return Result / Error
        alt Tool Execution Succeeded
            Tool-->>Orch: Structured Observation
            Orch->>Ollama: Next ReAct Turn (Evaluate Observation)
            Ollama-->>User: Final Conversational Answer
        else Tool Execution Failed (Closed-Loop Recovery)
            Tool-->>Orch: [Tool Error] + Recovery Hint + Allowed Actions
            Orch->>Ollama: Trigger Actuator Recovery Turn (Max Depth 5)
        end
    end
```

### Step-by-Step Breakdown:

1. **Fast-Path Interception ([`zelyr/core/fast_path_router.py`](../zelyr/core/fast_path_router.py)):**  
   Evaluates proposed actions against regex patterns and known binary maps (`KNOWN_APP_BINARIES`). Standard actions like launching apps (`open terminal`), navigating URLs (`open https://...`), or restarting systemd units are routed directly to native CLI commands (`kitty`, `xdg-open`, `systemctl`) in sub-10ms with zero model tokens expended.
2. **Window Context Introspection ([`zelyr/agents/native_orchestrator.py`](../zelyr/agents/native_orchestrator.py#L82-L134)):**  
   Non-blockingly queries `hyprctl activewindow -j` (0.5s timeout). Prepend telemetry headers:  
   `[Active Window: title="...", class="...", workspace=1]` to the initial user turn.
3. **Actuator Tier Arbitration ([`zelyr/agents/actuator_arbiter.py`](../zelyr/agents/actuator_arbiter.py)):**  
   Evaluates query intent and window context to select one of:
   - `tier1_ipc`: Window, process, clipboard, journal, volume, or network commands.
   - `tier2_browser`: Active browser windows, URLs, web searches, or DOM interactions.
   - `tier3_vision`: Non-accessible canvas applications, legacy games, or general UI fallback.
4. **Tool Schema Filtering ([`zelyr/core/plugins.py`](../zelyr/core/plugins.py#L598-L630)):**  
   Prunes unneeded schemas from the LLM prompt. Tier 1 queries expose only 9 core and OS tools; Tier 2 queries expose 5 core and web tools.
5. **Strict Negative Gating:**  
   If an active browser window is focused, [`zelyr/core/plugins.py`](../zelyr/core/plugins.py#L626-L627) strictly purges Tier 3 generic desktop vision tools (`interact_with_ui`) to prevent the model from guessing pixel coordinates on interactive web pages.
6. **Sliding-Window Context Pruning ([`zelyr/agents/native_orchestrator.py`](../zelyr/agents/native_orchestrator.py#L167-L208)):**  
   Enforces a sliding window keeping system directives, the first user prompt, and the most recent conversation turns within `max_context_tokens - 1000` tokens.
7. **Model Event Streaming ([`zelyr/agents/native_orchestrator.py`](../zelyr/agents/native_orchestrator.py#L381-L450)):**  
   Streams responses from Ollama over HTTP (`/v1/chat/completions`). Supports reasoning models by extracting `<think>` tags and streaming reasoning traces.
8. **Tolerant Tool Call Extraction ([`zelyr/agents/native_orchestrator.py`](../zelyr/agents/native_orchestrator.py#L224-L380)):**  
   Smaller local models (e.g., Qwen-7B) often emit concatenated JSON, markdown blocks, or reasoning text before tool invocations. The extractor parses raw text iteratively using `json.JSONDecoder().raw_decode` starting at candidate `{` tokens.
9. **Bounded Execution & Timeouts:**  
   Tools are executed with tier-specific timeouts (Tier 1: 10s–30s; Tier 2: 10s; Tier 3 Vision: 35s). Timeouts raise structured recovery hints instead of crashing the loop.
10. **Closed-Loop Error Recovery:**  
    If a tool fails, the error message and remedy hint (`[Tool Error]: ... Recovery Hint: ...`) are fed into the next turn. Conversational excuses are blocked, forcing the model to invoke the recovery tool directly.

---

## 3. The Three-Tier Actuator Model `[VERIFIED]`

Workstation automation is partitioned into three tiers to guarantee precision and avoid hallucinated mouse clicks:

| Tier | Subsystem | Verified Backing Tools | Latency | Precedence |
| :--- | :--- | :--- | :---: | :---: |
| **Tier 1** | **Deterministic IPC & Linux OS** | `manage_desktop_window`, `manage_system_process`, `manage_system_clipboard`, `query_system_journal`, `manage_media_playback`, `manage_workspace_file`, `send_desktop_notification`, `inspect_network`, `execute_command` | < 50ms | **Highest** |
| **Tier 2** | **Semantic Browser DOM & Web Search** | `interact_with_browser` (WebExtension WebSocket bridge at `ws://127.0.0.1:41144`), `web_search`, `fetch_web_page` | 100–300ms | **Intermediate** |
| **Tier 3** | **Visual Grounding Fallback** | `interact_with_ui`, Set-of-Mark (SoM) bounding boxes, OCR, and VLM click coordinates | 1000–3000ms | **Lowest** (Fallback) |

---

## 4. Context Budgeting & Sliding Window Algorithm `[VERIFIED]`

### A. Authoritative Precedence Hierarchy

The context token budget (`max_context_tokens`) is resolved dynamically with strict 4-level precedence:

```
[1. Explicit Caller Override]
          │ (e.g., caller_override=32768 in NativeOrchestrator payload)
          ▼
[2. Environment Variable]
          │ (ZELYR_NUM_CTX="16384")
          ▼
[3. User Configuration File]
          │ ("ollama_num_ctx": 8192 in ~/.config/zelyr/config.json)
          ▼
[4. System Default Fallback]
            (8,192 tokens)
```

### B. Context Validation & Clamping (`validate_and_clamp_num_ctx`)

- **Valid Range:** `[1024, 65536]`.
- Non-integers, booleans, negative numbers, or values outside `[1024, 65536]` log a warning and fall back to `8192`.
- Prevents GPU VRAM exhaustion on consumer hardware (e.g., RTX 4060 8GB).

### C. Sliding Window Message Pruning (`prune_messages`)

When total message tokens exceed `max_context_tokens - 1000`:
1. System directives are always retained.
2. The initial conversation turn (`conv_messages[0]`) is always retained to preserve original intent.
3. Older intermediate turns are dropped while the newest turns are retained up to the token budget.

---

## 5. Swarm Blackboard & Distributed Memory `[VERIFIED]`

The multi-agent swarm architecture uses an isolated blackboard store for inter-agent collaboration:

- **`BlackboardStore` ([`zelyr/memory/blackboard.py`](../zelyr/memory/blackboard.py)):**  
  Manages key-value state indexed by `session_id` and `agent_id`.
- **Namespace Isolation:**  
  Agents inside the same session cannot read or overwrite another agent's private keys unless shared via explicit consensus proposals.
- **Session Isolation:**  
  Concurrent sessions are completely isolated in memory.
- **Transaction Rollback:**  
  [`zelyr/core/transaction.py`](../zelyr/core/transaction.py) coordinates with `BlackboardStore`. Rolling back a transaction automatically purges associated blackboard state.

---

## 6. Dual-Orchestrator Comparison `[VERIFIED]`

The repository contains two orchestrator implementations serving distinct operational roles:

| Dimension | `NativeOrchestrator` ([`zelyr/agents/native_orchestrator.py`](../zelyr/agents/native_orchestrator.py)) | `OrchestratorAgent` ([`zelyr/agents/orchestrator_agent.py`](../zelyr/agents/orchestrator_agent.py)) |
| :--- | :--- | :--- |
| **Status** | **[VERIFIED] Active production CLI engine** | **[PARTIALLY VERIFIED] Extended state-machine engine** |
| **Entry Point** | Used by `zelyr`, `zelyr repl`, `zelyr chat` | Used by multi-agent swarm services |
| **Execution Loop** | Lightweight streaming async generator (`generate_stream`) | Explicit state machine (`THINK`, `ACT`, `OBSERVE`, `REFLECT`) |
| **Context Management** | Dynamic sliding-window pruning (`prune_messages`) | `ContextManager` token estimation |
| **Tool Loading** | Dynamic plugins via `zelyr.core.plugins` | Static registry via `zelyr.engine.tool_pruner` |
| **Ollama Communication**| Direct HTTP/SSE stream with `<think>` tag extraction | `SmartRouter` / `LiteLLM` abstraction |

# Zelyr Quickstart Guide

This guide provides reproducible, step-by-step instructions to install, configure, verify, and run Zelyr on a clean Linux workstation.

---

## Prerequisites

Before starting, ensure your system meets the following requirements:

| Component | Minimum Requirement | Recommended |
| :--- | :--- | :--- |
| **Operating System** | Linux (Kernel 6.x+) | Fedora 41/42/43 Atomic, Bazzite, Ubuntu 24.04 LTS, Arch Linux |
| **Display Server** | Linux Session (X11 or Wayland) | Wayland Compositor (Hyprland recommended for Tier 1 IPC) |
| **Python** | Python 3.10 or 3.11 | Python 3.11 via `uv` |
| **GPU / Acceleration** | CPU (AVX2 supported) | NVIDIA RTX 3060 / 4060 (8GB+ VRAM) or AMD ROCm |
| **Package Manager** | `pip` or `uv` | [`uv`](https://docs.astral.sh/uv/) (Astral) |
| **Inference Daemon** | Local inference engine | [Ollama](https://ollama.com/) 0.4.x+ |

---

## Step 1: Clone Repository & Install Dependencies

Zelyr uses `uv` for reproducible, isolated environment management.

### 1. Install `uv` (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or source ~/.zshrc
```

### 2. Clone the Repository
```bash
git clone https://github.com/Rutytoi220/Zelyr.git
cd Zelyr
```

### 3. Synchronize Dependencies
Install all core dependencies directly into a managed virtual environment:
```bash
uv sync
```

To include optional dependencies (e.g. PySide6 GUI, audio TTS/STT, or browser automation):
```bash
# Optional: install all GUI and automation dependencies
uv sync --extra automation
```

---

## Step 2: Configure Ollama & Pull Model Weights

Zelyr communicates with local LLMs over loopback (`http://127.0.0.1:11434`).

### 1. Install & Start Ollama
If Ollama is not installed:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify that the Ollama service is active:
```bash
curl -s http://127.0.0.1:11434/api/tags
```
If not running as a systemd service, start the daemon in a background shell:
```bash
ollama serve > /tmp/ollama.log 2>&1 &
```

### 2. Pull Recommended Models
Pull the target models based on your hardware profile:

```bash
# Primary Reasoning & Tool Dispatch (Recommended for 8GB+ VRAM)
ollama pull qwen3:8b

# Fast / Lightweight Reasoning (Low memory / CPU fallback)
ollama pull qwen3:1.7b

# Semantic Memory Embeddings (Required for vector memory)
ollama pull nomic-embed-text
```

---

## Step 3: Run Pre-Flight Evaluation & Verification

Before launching the interactive interfaces, run the self-contained verification suite to confirm that environment paths, database storage, and tool dispatch work cleanly.

### 1. Run Offline Pre-Flight Test
```bash
uv run python3 -m unittest -v test_install_bootstrap.py
```
This tests:
- Clean first-run auto-creation of `~/.config/zelyr/` directories (`tools.d`, `plugins`, `sessions`, `logs`).
- Zero hardcoded developer paths in production code.
- Package entry points and dependencies.

### 2. Run Level 1 Deterministic Benchmark Suite
```bash
uv run python3 -m zelyr.eval --level 1
```
Expected output:
```text
Executing 6 benchmark task(s)...
[1/6] Running task_tool_dispatch (Level 1)... PASS
[2/6] Running task_recovery_envelope (Level 1)... PASS
[3/6] Running task_step_limit_enforcement (Level 1)... STEP_LIMIT_EXCEEDED
[4/6] Running task_false_success_rejection (Level 1)... FALSE_SUCCESS
[5/6] Running task_memory_persistence_roundtrip (Level 1)... PASS
[6/6] Running task_timeout_containment (Level 1)... TIMEOUT

Total: 6 | Pass: 3 | False Success: 1 | Fail: 0 | Step Limit: 1 | Timeout: 1
```

*(Runs 100% offline in under 3 seconds with zero network or GPU dependencies).*

---

## Step 4: Launch Interactive Interfaces

Zelyr provides multiple interfaces designed for different workflows:

### A. Modern Streaming REPL (Default)
The Tokyo Night inline streaming REPL provides interactive multi-turn conversations and real-time tool telemetry:
```bash
uv run python3 -m zelyr.cli.main
# or explicitly:
uv run python3 -m zelyr.cli.main repl
```

Useful slash commands inside the REPL:
- `/help`: Display available REPL commands.
- `/model`: Switch the active reasoning model.
- `/tools`: List registered tools in the Three-Tier hierarchy.
- `/memory list`: View persistent facts in the SQLite store.
- `/doctor`: Run diagnostic health checks across all subsystems.
- `/exit`: Exit the session cleanly.

### B. Full-Screen Terminal UI (TUI)
For an IDE-like terminal workspace:
```bash
uv run python3 -m zelyr.cli.main chat
```

### C. Desktop GUI (PySide6)
For graphical window management and visual diagnostics:
```bash
uv run python3 -m zelyr.cli.main ui
```

### D. Single-Shot Natural Language Task Execution
To execute an autonomous task from the shell:
```bash
uv run python3 -m zelyr.cli.main run "List open windows and create a summary in my workspace"
```

---

## Step 5: Troubleshooting & Hardware Sizing

### 1. GPU VRAM Out-of-Memory (CUDA OOM)
- **Symptom:** Ollama emits `HTTP 500: CUDA out of memory` or crashes during generation.
- **Root Cause:** Deep reasoning models (`qwen3:8b`) have a ~5.2 GB model footprint. Allocating a 32K context window (`num_ctx: 32768`) creates an additional ~4.83 GB KV cache buffer, which exceeds 8GB of VRAM (5.2 GB + 4.83 GB = 10.03 GB).
- **Solution:** Zelyr unifies and clamps the default context budget to **8,192 tokens** (`num_ctx: 8192`), reducing KV cache allocation to ~1.0 GB and keeping total GPU usage under ~6.5 GB.
- **Override Context Budget:** If running on a GPU with 16GB+ VRAM or using a smaller model, adjust context dynamically via environment variable:
  ```bash
  export ZELYR_NUM_CTX=16384
  ```
  or configure `ollama_num_ctx` in `~/.config/zelyr/config.json`.

### 2. Ollama Backend Unreachable
- **Symptom:** `⚠️ [FastAPI Error] Could not reach Ollama backend: Connection refused`.
- **Solution:** Verify that the daemon is listening on port 11434:
  ```bash
  curl http://127.0.0.1:11434/api/tags
  ```
  If Ollama is listening on a custom address or container host, specify it via:
  ```bash
  export OLLAMA_BASE_URL="http://127.0.0.1:11434"
  ```

### 3. Hyprland IPC Commands Fail Outside Hyprland
- **Symptom:** `manage_desktop_window` returns `hyprctl: command not found` or fails to query active windows.
- **Solution:** Hyprland IPC tools require a running Hyprland compositor session (`$HYPRLAND_INSTANCE_SIGNATURE`). On standard GNOME, KDE, or headless environments, Zelyr automatically degrades to generic OS utilities (`wmctrl`, `xdotool`, or keyword fallback).

### 4. Running Live Benchmarks
To benchmark your local model against the Level 2 tool-dispatch test battery:
```bash
uv run python3 -m zelyr.eval --level 2 --live -m qwen3:8b --json my_benchmark_report.json
```
For detailed benchmark methodology and empirical baseline figures, refer to [`docs/benchmarks/live_baseline_report.md`](benchmarks/live_baseline_report.md).

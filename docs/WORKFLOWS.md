# Zelyr Operational Workflows Guide

This guide documents source-verified end-to-end workflows for configuring, validating, running, auditing, and recovering **Zelyr** (v1.4.1). All commands and flags match `zelyr --help` and active modules in [`zelyr/`](../zelyr/).

---

## 1. Installation & First Launch

```bash
git clone https://github.com/Rutytoi220/Zelyr.git
cd Zelyr
uv sync
```

First CLI invocation calls `zelyr.config.initialize_directories()` and provisions:
- `~/.config/zelyr/tools.d/` (dynamic tool plugins)
- `~/.config/zelyr/plugins/` (extension plugins)
- `~/.config/zelyr/sessions/`, `logs/`, and `models/`

Start Zelyr interactively or via subcommands:

```bash
uv run zelyr                  # Modern inline streaming REPL (default, alias: zelyr repl)
uv run zelyr chat             # Multi-turn REPL with session flags (--session, --new)
uv run zelyr ui               # PySide6 desktop GUI (alias: zelyr gui)
uv run zelyr status           # System, configuration, and daemon status
```

---

## 2. Configuration & Precedence Diagnostics

Configuration lives in `ZELYR_CONFIG_DIR/config.json` (default `~/.config/zelyr/config.json`, falling back to legacy AXIOM paths only if Zelyr paths are absent).

Effective precedence for core runtime knobs resolves top-down:
1. **Explicit caller argument** (for example `zelyr run --base-url ...`)
2. **Environment variable (`ZELYR_*`, then legacy `AXIOM_*`)**
3. **Config file (`config.json`)**
4. **Built-in `ZelyrConfig` defaults**

Key environment variables honored across runtime subsystems (`zelyr/config.py`):
- `ZELYR_OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`)
- `ZELYR_NUM_CTX` (clamped `[1024, 65536]`, default `8192`)
- `ZELYR_OLLAMA_KEEP_ALIVE` (default `"5m"`)
- `ZELYR_OLLAMA_READ_TIMEOUT` (clamped `[5.0, 3600.0]`, default `300.0`)
- `ZELYR_AUTOPILOT` / `ZELYR_TESTING` (`"1"` enables non-interactive execution bypasses where documented)
- `ZELYR_CONFIG_DIR` (overrides config root directory)

Inspect effective configuration sources, legacy key migrations, unknown JSON keys, env shadowing, and type/range validation before launching heavy workloads:

```bash
zelyr config show [--config-path PATH] [--json]
zelyr config validate [--config-path PATH] [--strict] [--json]
```

`validate` / `--strict` exits non-zero if JSON parsing fails, values fail validation, or unknown keys are present. `zelyr/engine/config_diagnostics.py` scrubs secret values, tokens, and embedded URL credentials from outputs.

---

## 3. Model & Provider Compatibility Checks

Evaluate whether the configured provider endpoint, installed model, context sizing (`ollama_num_ctx`), and capabilities are actually verified for a Zelyr workflow:

```bash
# Live endpoint metadata check (/api/tags + /api/show on Ollama, or /v1/models on OpenAI-compatible)
zelyr model check [--model MODEL] [--base-url URL] [--workflow planned_graph|react_tools|chat|vision] [--json]

# Offline inspection without HTTP calls (reports UNKNOWN rather than guessing from model name)
zelyr model check --no-probe --json

# Combine runtime environment manifest + model compatibility check
zelyr env --model-check [--no-probe] [--json]
```

Status semantics (`CapabilityStatus`):
- `AVAILABLE`: Verified directly from provider/protocol metadata (`source: detected` or `protocol_probe`).
- `DEGRADED`: Endpoint reachable but capability degraded (e.g., model missing from catalog, `num_ctx` exceeds model `context_length`, native `tools` missing so `FallbackToolParser` is used, or non-200 HTTP auth error).
- `UNAVAILABLE`: Provider unreachable/timed out, invalid workflow requirement, or failed protocol fixture.
- `UNKNOWN`: Not probed or cannot be verified offline (e.g., `multi_turn_tool_calling` and `structured_output_support` without live execution evidence). Capabilities are **never** inferred solely from a model's name.

Use `--strict` to exit non-zero whenever `overall_status != "AVAILABLE"` or `workflow_compatible` is not `True`.

---

## 4. Unified Execution Policy & Confirmations

Every tool dispatch path (`NativeOrchestrator` ReAct loop, parallel `TaskGraph`, retries/recovery, `ToolRegistry`, `zelyr tool run`, external/MCP adapters, and fast-path routing) goes through two ordered security boundaries:
1. **`authorize_and_validate_tool_call` ([`zelyr/tool_registry.py`](../zelyr/tool_registry.py))**: canonicalizes tool aliases, strips untrusted model keys, checks `allowed_tools` and privilege rings (`0 <= caller_ring <= required_ring <= 3`), and validates argument schemas.
2. **`ExecutionPolicyEngine` ([`zelyr/engine/execution_policy.py`](../zelyr/engine/execution_policy.py))**: classifies action risk (`READ_ONLY`, `WORKSPACE_MUTATING`, `ELEVATED`, `DESTRUCTIVE` / irreversible) against effective `execution_policy_mode` (`conservative`, `strict`, `balanced`, `basic`, `autopilot`), resource containment rules, and single-use `ConfirmationStore` bindings.

### Inspecting Policy & Previewing Without Executing

```bash
# View effective policy mode, precedence, risk defaults, and overrides
zelyr policy show [--policy-mode MODE] [--config-path PATH] [--json]

# Dry-run explanation (preview=True never mutates state or consumes confirmations)
zelyr policy explain manage_workspace_file \
  --args '{"action": "write", "path": "notes.txt", "content": "hi"}' \
  --caller-ring 2 [--json]

# Query redacted AUTHORIZATION_DECISION history from the execution ledger
zelyr policy history [--limit 20] [--outcome allow|require_confirmation|deny] [--tool TOOL] [--execution-id ID] [--json]
```

Inside the interactive REPL, run `/policy show`, `/policy explain <tool_id>`, or `/policy history`.

### Bound Confirmation Guarantees
Confirmations issued by `ConfirmationStore` are single-use SHA-256-hashed tokens bound strictly to `(tool_id, arg_digest, resource_scope, execution_id, node_id, policy_version)`. If arguments mutate, the node is replanned to a different `node_id`, TTL expires, or a token is replayed, policy evaluation fails closed (`DENY`). Raw confirmation tokens are never stored in checkpoints, ledger events, or model context.

---

## 5. Planned Graphs & Bounded Parallel Execution

Convert a multi-step prompt into an explicit, validated DAG (`TaskGraph`) or run single-shot ReAct orchestration via `zelyr run`:

```bash
# Compile and inspect the TaskGraph plan without executing anything
zelyr run --plan-only "Read config.json and write a summary report"

# Compile and execute via TaskGraph with explicit confirmation, bounded parallelism, and outcome verification
zelyr run --plan --confirm --max-concurrency 2 --verify "file_exists:report.txt" \
  "Read config.json and write report.txt"

# Standard ReAct execution (one tool execution per model turn)
zelyr run --verify "file_exists:output.txt" "Create output.txt"
```

Key graph execution invariants (`zelyr/planning/compiler.py`, `zelyr/engine/task_graph.py`, `zelyr/agents/native_orchestrator.py`):
- **Default Sequential Safety:** `max_concurrency` defaults to `1`; values `<= 0` fail validation cleanly.
- **Resource Conflict Serialization:** Disjoint read-only or separate-target nodes execute concurrently up to `max_concurrency`, while overlapping file targets, parent/child directory containment, exclusive GUI/browser/shell profiles, and duplicate operation fingerprints serialize atomically.
- **Verification Gating:** Downstream nodes with `requires_verified_dependencies=True` remain blocked until upstream prerequisites reach `COMPLETED_VERIFIED`.
- **Untrusted Output Sanitization:** Model plans cannot smuggle concurrency overrides or policy bypass flags (`FORBIDDEN_PLAN_SECURITY_KEYS`, `FORBIDDEN_PLAN_CONCURRENCY_KEYS`).

---

## 6. Verification, Recovery, Checkpoints & Resume

Every execution is recorded in SQLite via `CentralExecutionLedger`, with per-step continuity snapshots saved through `CheckpointManager`:

```bash
# List recent executions (filter to interrupted/incomplete tasks)
zelyr executions [--resumable] [--status STATUS] [--limit 20] [--json]

# Inspect durable checkpoints
zelyr checkpoint list [--all] [--limit 20] [--json]
zelyr checkpoint inspect <execution_id> [--json]
zelyr checkpoint abandon <execution_id> [--reason "Superseded"] [--json]

# Inspect or safely resume an interrupted execution
zelyr resume --list
zelyr resume <execution_id> --inspect
zelyr resume <execution_id> --confirm [--allow-unverified-retry] [--verify CHECK:TARGET[:EXPECTED]]
```

Safety rules during recovery & resume (`zelyr/engine/checkpoint.py`, `zelyr/engine/recovery.py`):
- **Skip Verified Work:** Operations confirmed `COMPLETED_VERIFIED` are not re-executed.
- **Block Blind Mutating Replay:** Mutating operations interrupted mid-flight, timed out, or cancelled after dispatch cannot blindly auto-retry; they require host state reconciliation (`OutcomeVerifier`) or explicit operator confirmation (`--allow-unverified-retry`).
- **Live Re-Authorization:** Resumed tasks always re-check current caller permissions, tool availability, environment drift, and `ExecutionPolicyEngine` rules rather than trusting historical grants.

---

## 7. Lifecycle Diagnostics & Observability Metrics

Run read-only, redacted post-mortem diagnostics or inspect measured latency/count telemetry:

```bash
# End-to-end execution diagnosis (request -> plan/graph -> nodes -> tools -> policy -> verification -> recovery -> blockers)
zelyr diagnose <execution_id> [--verify] [--json]
# Equivalent flag on execution inspector:
zelyr execution <execution_id> --checkpoint --graph --diagnose [--verify] [--metrics]

# Inside REPL:
# /diagnose <execution_id> [--json]

# Measured timings and event counts (single execution or fleet summary)
zelyr metrics [<execution_id>] [--limit 20] [--json]   # alias: zelyr observability
```

`zelyr metrics` strictly separates `measurement_kind: "measured"` monotonic timings (`model_requests`, `tool_duration_ms`, `verification_duration_ms`, `confirmation_wait_ms`) from unmeasured legacy or absent provider metrics (`token_usage` listed under `unmeasured_metrics` when omitted by the provider).

---

## 8. External Tools & Model Context Protocol (MCP)

Zelyr supports both consuming external MCP servers and exposing Zelyr tools as an MCP gateway:

- **Consuming MCP Servers ([`zelyr/mcp/manager.py`](../zelyr/mcp/manager.py), [`zelyr/tool_registry.py`](../zelyr/tool_registry.py)):**
  Configured in `ZELYR_CONFIG_DIR/mcp.json` (`{"mcpServers": { ... }}`) supporting `stdio` (`command`, `args`) and `sse` (`url`, `headers`) transports.
  Newly discovered external tools default to **unauthorized** and require explicit host authorization (`ToolRegistry.authorize_external_tool`) before registration/dispatch. Every external call records `is_external`, `source_id`, and `source_type` telemetry in the execution ledger.
- **Serving Zelyr Over MCP:**
  ```bash
  zelyr mcp serve --transport stdio --caller-ring 1 [--allowed-tools tool1,tool2] [--no-memory]
  # Or HTTP/SSE bound locally:
  zelyr mcp serve --transport sse --host 127.0.0.1 --port 8000 --caller-ring 1
  ```

---

## 9. Troubleshooting Common Operational Issues

| Symptom / Failure | Diagnostic Command | Cause & Resolution |
| :--- | :--- | :--- |
| Config warnings / unexpected overrides | `zelyr config validate --strict` | Env vars (`ZELYR_*`) override `config.json`; inspect `validation_issues` and remove unknown/obsolete keys. |
| Provider unreachable or missing model | `zelyr model check` | Start Ollama (`ollama serve`), verify `ollama list` / pull model, or pass valid `--base-url` (`http`/`https`). |
| Model tool errors or context truncation | `zelyr model check --workflow planned_graph` | Ensure model `context_length >= configured num_ctx` (`ZELYR_NUM_CTX`), or reduce `ollama_num_ctx` to `8192` to avoid GPU OOM. |
| Tool call blocked (`REQUIRE_CONFIRMATION` / `DENY`) | `zelyr policy explain <tool> --args '{...}'` | Inspect `reason_code` / `determining_rule_id`. For `--plan` workflows pass `--confirm`; in headless non-TTY runs explicit `ZELYR_AUTOPILOT=1` or confirmation is required. |
| Interrupted task refuses auto-resume | `zelyr resume <id> --inspect` & `zelyr diagnose <id> --verify` | Cancelled tasks require `--confirm`; ambiguous post-dispatch mutating operations require state reconciliation or explicit `--allow-unverified-retry`. |
| Suspected false success / missing output | `zelyr diagnose <id> --verify` | Compares model/tool reported output against independent `OutcomeVerifier` filesystem/state evidence and highlights `reported_only_unverified`. |

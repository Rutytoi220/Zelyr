# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0-rc.1] - 2026-10-28

Release Candidate 1 for Zelyr v1.1.0, delivering end-to-end security hardening, resilient OTA updates, authenticated IPC, autonomous multi-turn evaluation benchmarks, and distribution builds.

### Highlights & Architecture Pillars

- **Pillar 1: Ring-0 AST Validation & Transport-Level SSRF Protection**
  - Integrated `SecurityASTValidator` inspecting dynamic Python code for dangerous AST nodes, forbidden imports (`os`, `subprocess`, `sys`), and dynamic code execution (`eval`, `exec`).
  - Added strict Ring-0 authorization enforcement in `ToolRegistry` preventing dynamic tools from claiming Ring 0 or mutating core tools.
  - Implemented transport-level SSRF defense with strict IP pinning, defeating DNS rebinding attacks and validating against loopback, private, link-local, multicast, and cloud metadata endpoints.
- **Pillar 2: Resilient OTA Auto-Updater & Daemon Self-Healing Supervisor**
  - Implemented atomic staging pipeline (`staging_<version>`) with SHA256 verification, pre-flight dry-run health checks, and automated directory rollback on failure.
  - Built `DaemonWorkerSupervisor` featuring exponential backoff (1s -> 30s) and a 5-crash circuit breaker to contain repeated process or task crashes.
  - Hardened daemon shutdown sequence with SIGTERM, bounded 5s wait, and SIGKILL escalation.
- **Pillar 3: Authenticated IPC Protocol, Rate Limiting & Frame Ceilings**
  - Enforced POSIX 0600 cryptographic token authentication (`daemon.token`) across WebSocket connections via bearer headers or query tokens.
  - Established strict JSON-RPC 2.0 Pydantic schema validation with standard error codes (-32700, -32600, -32601, -32029).
  - Implemented 1MB payload ceilings (`PayloadTooLargeError`, code 4409) and sliding window rate limiting (50 req/s).
- **Pillar 4: Autonomous Multi-Turn Benchmark Expansion (24/24 Tasks)**
  - Expanded benchmark evaluation suite from 18 to 24 tasks with the introduction of Level 4 autonomous self-repair and resilient subsystems.
  - Tasks include runtime bug patching, Swarm blackboard coordination, corrupted state rollback, ETL pipelines, sandbox containment, and proactive memory decay.
  - Achieved a 100.0% offline pass rate across all 24 tasks in ~1.16s execution time.
- **Pillar 5: Multi-Platform Distribution Build Verification**
  - Verified wheel and sdist package builds from clean isolated source trees with `scripts/build_dist.py`.
  - Version alignment synchronized to `1.1.0rc1` across Python packages, Web UI (`1.1.0-rc.1`), and browser extension manifest (`1.1.0.1`).

### Distribution Release Assets

| Artifact Name | Platform / Type | SHA256 Checksum | Size |
| :--- | :--- | :--- | :--- |
| `zelyr_core-1.1.0rc1-py3-none-any.whl` | Python Wheel (`any`) | `ac470442195d1e2a8499b79570486c99ed6f4486a23096fd6ae77133397b2d0e` | 2.1 MiB |
| `zelyr_core-1.1.0rc1.tar.gz` | Source Distribution (`sdist`) | `39468d09336782edb3ff951f28aba409bcc9efc6b1c5482967a454f2c97f3851` | 2.0 MiB |

### Added
- **Deterministic Offline Evaluation Benchmarks:** Full 24/24 offline benchmark execution via deterministic ReAct mock queues for Levels 1, 2, 3, and 4.
- **Standardized Optional Dependency Boundary Guards:** Added `OptionalDependencyError`, zero-import `is_extra_available`, and `require_extra` helpers in `zelyr.utils.optional_deps` for robust degradation when optional extras are missing.
- **Structured Tool Failure Envelopment:** Integrated structured `ToolResult` failure envelopment into `ToolRegistry` to safely report missing optional dependencies and prevent agent ReAct loops from crashing.
- **Hardened Swarm Multi-Agent Consensus Quorum & Rollback:** Hardened $\ge 2/3$ quorum evaluation against voter timeouts and exceptions, added automatic ephemeral Blackboard point-in-time snapshot rollback upon rejected proposals, and implemented fail-safe diagnostic logging for split-decision ties.

### Performance
- **PEP 562 Lazy Module Exports:** Implemented PEP 562 package-level lazy exports across `zelyr/__init__.py`, `axiom/__init__.py`, and `zelyr/main.py`, reducing root import module count from 4,322 to 77 and cutting CLI cold-start latency from ~2.7s to <90ms (32x faster).
- **Non-Blocking Telemetry & Bounded Probing:** Converted `zelyr status` to non-blocking kernel load sampling and a 20ms bounded socket probe, cutting execution latency to ~83ms (~2.7x faster).
- **Deferred Plugin Discovery:** Deferred `PluginManager` initialization and filesystem discovery in `zelyr tool list` behind the `--all`/`--plugins` flag, reducing tool listing latency to ~133ms (~2.3x faster).

## [1.0.0] - 2026-10-28

Landmark official release of Zelyr, establishing a unified, secure, and resilient local-first autonomous AI agent ecosystem.

### Highlights & Architecture

- **Namespace Transition:** Full migration from legacy AXIOM to Zelyr across core modules, CLI entry points (`zelyr`), and standard system directories (`~/.config/zelyr`, `~/.local/share/zelyr`), complete with backward-compatibility runtime shims and entrypoint aliases.
- **Modular Dependencies:** Decoupling of heavy machine learning (`torch`, `transformers`) and browser automation (`playwright`) dependencies into optional extras (`[ml]`, `[browser]`), streamlining the base installation footprint.
- **Memory Subsystem:** High-performance SQLite WAL storage engine featuring FTS5 full-text search indexing, semantic deduplication, and automated background REM sleep memory compaction and fact promotion.
- **Security Perimeter:** Hardened defense-in-depth architecture featuring a Ring-0 AST validator for command and script inspection, strict SSRF input normalization and private IP containment, and protected core tool registries.
- **Swarm Triad:** Autonomous multi-agent consensus engine pairing Supervisor, Coder, and TestRunner agents with $\ge 2/3$ quorum voting, blackboard shared memory, and transactional state rollbacks.

### Published Distribution Artifacts

The official v1.0.0 release includes 7 multi-platform distribution assets attached to GitHub Release [v1.0.0](https://github.com/Rutytoi220/Zelyr/releases/tag/v1.0.0):

| Artifact Name | Platform / Type | SHA256 Checksum | Size |
| :--- | :--- | :--- | :--- |
| `zelyr_core-1.0.0-py3-none-any.whl` | Python Wheel (`any`) | `bf65138e9bf4c3b775ba0457771b29044bf0aee27e1ab0ffc1b9c38b168387a7` | 2.05 MiB |
| `zelyr_core-1.0.0.tar.gz` | Source Distribution (`sdist`) | `62273d608803d2c7aa8375eb12400c526808b8431bbb8e99948b6cc5a5d08a00` | 1.91 MiB |
| `zelyr_1.0.0_amd64.deb` | Debian / Ubuntu Package (`amd64`) | `fe45d4823fa5c7293b2410a59881911b36f512e0e941f0b7db7a815bf5dd5134` | 452.36 MiB |
| `zelyr-1.0.0-1.x86_64.rpm` | Fedora / RHEL Package (`x86_64`) | `0944b4f2d4b660fa21f57e74fed30095f7f307497ca7b7e3c35cfd404e878385` | 452.56 MiB |
| `Zelyr-v1.0.0-x86_64.AppImage` | Standalone Linux AppImage (`x86_64`) | `31cb89f9dddd4874bd516c4091ca0702bd94826e84072876c7bdfe88bc829a11` | 400.13 MiB |
| `zelyr_latest.dmg` | macOS Disk Image (`dmg`) | `46bea3ad2868ac3f5ae920e6cb4e81f4dda26a89b17062d724b86d849ab452ab` | 815.88 MiB |
| `Zelyr.exe` | Windows Standalone Executable (`x86_64`) | `c5a783b9df6afb48d31da83876208e14359d22043080491192bd6018ed99a892` | 48.52 MiB |

## [v11.3.0] - Production Hardening & CI/CD
This release hardens the build system, wires in OTA capabilities, and eliminates UI teardown aborts for a pristine out-of-the-box experience.

### Added
- **Declarative Theme Engine:** Dynamic JSON-based theme registry loaded effortlessly into PySide6 `QSS` layouts without hardcoded hex colors.
- **OTA Updater:** Implemented GitHub Release polling in `axiom/network/updater.py` to prompt users when a newer `AppImage` tag drops.
- **OOBE Wizard:** Full Out-Of-Box Experience (`axiom/gui/widgets/oobe_wizard.py`) providing a guided setup for Local LLMs on first boot.
- **Automated CI/CD:** GitHub Actions `release.yml` now runs headless PySide6 test suites (`QT_QPA_PLATFORM=offscreen`) and automatically builds/attaches the AppImage for tagged releases.

### Fixed
- **Headless PySide6 Test Architecture:** Resolved fatal C++ `SIGABRT` crashes on PyInstaller teardown by systematically yielding and `quit()`ing `QThread` instances within `AxiomHubDialog`.
- **AppImage Build Resilience:** Fortified `scripts/build_appimage.sh` with correct `usr/share/applications` AppDir directory structure and explicit `set -e` failure trapping.
- **Circuit Breakers:** `SystemHealthWatchdog` now survives nested permission errors silently rather than stalling the central EventBus.

[Unreleased]: https://github.com/Rutytoi220/Zelyr/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Rutytoi220/Zelyr/releases/tag/v1.0.0
[v11.3.0]: https://github.com/Rutytoi220/Zelyr/releases/tag/v11.3.0

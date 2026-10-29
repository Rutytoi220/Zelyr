# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-10-29

General Availability (GA) official production release of Zelyr v1.1.0, delivering comprehensive execution security hardening, sub-100ms CLI startup latency, modular optional dependency boundaries, resilient atomic OTA updates, authenticated WebSocket IPC, Swarm consensus quorum resilience, REM sleep memory consolidation, and full 24/24 autonomous multi-turn evaluation benchmarks.

### Highlights & Architecture Pillars

- **Pillar 1: Sub-100ms CLI Startup & Non-Blocking Telemetry**
  - Implemented PEP 562 package-level lazy exports across `zelyr/__init__.py`, `axiom/__init__.py`, and `zelyr/main.py`, reducing root import module count from 4,322 to 77 and cutting cold-start latency from ~2.7s to ~67ms (32x faster).
  - Converted `zelyr status` to non-blocking kernel load sampling and a 20ms bounded socket probe (~83ms runtime).
  - Deferred `PluginManager` initialization and filesystem discovery in `zelyr tool list` behind `--all`/`--plugins` flags.
- **Pillar 2: Modular Dependency Boundaries & Structured Degradation**
  - Standardized `OptionalDependencyError`, zero-import `is_extra_available`, and `require_extra` helpers in `zelyr.utils.optional_deps` for robust degradation when optional extras are missing.
  - Decoupled optional extras (`gui`, `server`, `vision`, `audio`, `dev`) with actionable remediation instructions (`pip install 'zelyr-core[gui]'`).
  - Integrated structured `ToolResult` failure envelopment into `ToolRegistry` to safely report missing optional dependencies and prevent agent ReAct loops from crashing.
- **Pillar 3: Ring-0 AST Validation & Transport-Level SSRF Defense**
  - Integrated `SecurityASTValidator` inspecting dynamic Python code for dangerous AST nodes, forbidden imports (`os`, `subprocess`, `sys`), and dynamic code execution (`eval`, `exec`).
  - Added strict Ring-0 authorization enforcement in `ToolRegistry` preventing dynamic tools from claiming Ring 0 or mutating core tools.
  - Implemented transport-level SSRF defense with strict IP pinning, defeating DNS rebinding attacks and validating against loopback, private, link-local, multicast, and cloud metadata endpoints.
- **Pillar 4: Resilient Atomic OTA Updates & Daemon Worker Circuit Breaking**
  - Implemented atomic staging pipeline (`staging_<version>`) with SHA256 verification, pre-flight dry-run health checks, and automated directory rollback on failure.
  - Built `DaemonWorkerSupervisor` featuring exponential backoff (1s -> 30s) and a 5-crash circuit breaker to contain repeated process or task crashes.
  - Hardened daemon shutdown sequence with SIGTERM, bounded 5s wait, and SIGKILL escalation.
- **Pillar 5: Authenticated WebSocket IPC with Pydantic Contracts and Rate Limiting**
  - Enforced POSIX 0600 cryptographic token authentication (`daemon.token`) across WebSocket connections via bearer headers or query tokens.
  - Established strict JSON-RPC 2.0 Pydantic schema validation with standard error codes (-32700, -32600, -32601, -32029).
  - Implemented 1MB payload ceilings (`PayloadTooLargeError`, code 4409) and sliding window rate limiting (50 req/s).
- **Pillar 6: Swarm Quorum Hardening & Ephemeral Blackboard Rollback**
  - Hardened $\ge 2/3$ quorum evaluation against voter timeouts and exceptions with fail-safe diagnostic logging for split-decision ties.
  - Added automatic ephemeral Blackboard point-in-time snapshot rollback upon rejected proposals.
- **Pillar 7: REM Sleep Memory Compaction, Temporal Decay & FTS5 Concurrency**
  - Verified and hardened SQLite WAL concurrency under simultaneous read/write multi-agent operations.
  - Integrated exponential temporal decay scoring and semantic deduplication during sleep cycles.
  - Enforced FTS5 full-text index synchronization and consistency during memory pruning and archival.
- **Pillar 8: Autonomous Multi-Turn Benchmark Expansion (24/24 Tasks, Levels 1-4)**
  - Expanded benchmark evaluation suite from 18 to 24 deterministic offline tasks covering Levels 1 through 4.
  - Validated Level 4 self-repair, Swarm coordination, atomic state rollback, ETL pipelines, tool sandboxing, and memory decay with 100.0% pass rate in ~1.18s.

### Distribution Release Assets

| Artifact Name | Platform / Type | SHA256 Checksum | Size |
| :--- | :--- | :--- | :--- |
| `zelyr_core-1.1.0-py3-none-any.whl` | Python Wheel (`any`) | `adf27e0648d9d7d9ab1a99d93b88293d993ce14932812efd07e37871ffe5de33` | 2.1 MiB |
| `zelyr_core-1.1.0.tar.gz` | Source Distribution (`sdist`) | `cfc7ee71e5582dc3344a88738e728a94e4c8b362093abb2f0f9b528faedde5a6` | 2.0 MiB |

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

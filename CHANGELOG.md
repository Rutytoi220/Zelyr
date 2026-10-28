# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Deterministic Offline Evaluation Benchmarks:** Full 18/18 offline benchmark execution via deterministic ReAct mock queues for live workspace diagnostic, metric calculation, and web research synthesis tasks (`zelyr/eval/tasks/model_tasks_p3.py`).

### Changed
- **Decoupled CI Release Publishing:** Decoupled GitHub Release creation from PyPI publishing in `.github/workflows/publish.yml` to allow standalone release asset generation and distribution independently of PyPI token availability.
- **Development Version Alignment:** Advanced development baseline to `1.1.0.dev0` across all metadata manifests, packages, and services.

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

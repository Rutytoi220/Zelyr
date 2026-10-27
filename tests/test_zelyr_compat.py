"""Compatibility checks for the AXIOM -> Zelyr identity migration."""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

from zelyr import compat
from zelyr.config import AxiomConfig, ZelyrConfig


def test_legacy_import_shim_and_class_aliases() -> None:
    import axiom
    import zelyr
    from axiom.config import AxiomConfig as LegacyConfig
    from axiom.core.engine import Engine
    from axiom.fs.axiom_fs import AxiomFS
    from axiom.plugins.axiom_plugin import AxiomPlugin as LegacyPlugin
    from zelyr.plugins.zelyr_plugin import ZelyrPlugin

    assert axiom.__version__ == zelyr.__version__
    assert axiom.Engine is zelyr.Engine
    assert Engine is zelyr.Engine
    assert LegacyConfig is ZelyrConfig
    assert AxiomConfig is ZelyrConfig
    assert LegacyPlugin is ZelyrPlugin
    assert AxiomFS.__name__ == "ZelyrFS"


def test_legacy_module_can_be_executed_with_dash_m() -> None:
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    process = subprocess.run(
        [sys.executable, "-m", "axiom.cli.main", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert "Zelyr" in process.stdout


def test_legacy_tool_decorator_and_marker_are_supported() -> None:
    from zelyr.tools.base import axiom_tool, zelyr_tool

    assert axiom_tool is zelyr_tool

    @axiom_tool("legacy_tool", "legacy description", {"type": "object"})
    def legacy_tool(**kwargs):
        return kwargs

    assert legacy_tool.__zelyr_tool__ is True
    assert legacy_tool.__axiom_tool__ is True
    assert legacy_tool.__tool_name__ == "legacy_tool"


def test_legacy_public_constant_aliases() -> None:
    from axiom.plugins.loader import AXIOM_RUNTIME_VERSION, ZELYR_RUNTIME_VERSION
    from axiom.services.ota_updater import AXIOM_VERSION, ZELYR_VERSION

    assert AXIOM_RUNTIME_VERSION == ZELYR_RUNTIME_VERSION
    assert AXIOM_VERSION == ZELYR_VERSION


def test_legacy_json_rpc_methods_and_event_dialect() -> None:
    import asyncio
    import json

    from zelyr.core.ipc_server import ZelyrDaemon

    assert compat.canonical_rpc_method("axiom.prompt") == "zelyr.prompt"
    assert compat.canonical_rpc_method("zelyr.prompt") == "zelyr.prompt"
    assert compat.rpc_method_prefix("axiom.status") == "axiom."
    assert compat.rpc_method_prefix("zelyr.status") == "zelyr."

    daemon = object.__new__(ZelyrDaemon)
    daemon.token = "session-token"
    daemon.subscribers = {}
    daemon._client_protocols = {}
    daemon.engine = SimpleNamespace(
        is_running=lambda: True,
        registry=SimpleNamespace(
            list_agents=lambda: {"agent": object()},
            list_tools=lambda: {"tool": object()},
        ),
    )
    daemon.cli = SimpleNamespace(
        ollama=SimpleNamespace(is_available=lambda: True),
        orchestrator=SimpleNamespace(
            run=lambda text: SimpleNamespace(success=True, output=f"Processed: {text}")
        ),
    )

    async def run() -> None:
        auth = await daemon._process_rpc_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "axiom.authenticate",
                    "params": {"token": "session-token"},
                    "id": 1,
                }
            ),
            False,
            False,
        )
        assert auth["result"] == "Authenticated"

        prompt = await daemon._process_rpc_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "axiom.prompt",
                    "params": {"text": "hello"},
                    "id": 2,
                }
            ),
            True,
            False,
        )
        assert prompt["result"] == "Processed: hello"

        status = await daemon._process_rpc_message(
            json.dumps({"jsonrpc": "2.0", "method": "axiom.status", "id": 3}),
            True,
            False,
        )
        assert status["result"] == {
            "engine_running": True,
            "llm_available": True,
            "agents": 1,
            "tools": 1,
        }

        class FakeWebSocket:
            def __init__(self) -> None:
                self.sent: list[str] = []

            async def send(self, message: str) -> None:
                self.sent.append(message)

        legacy = FakeWebSocket()
        canonical = FakeWebSocket()
        daemon.subscribers = {legacy: {"*"}, canonical: {"*"}}
        daemon._client_protocols = {legacy: "axiom.", canonical: "zelyr."}
        daemon._dispatch_event(SimpleNamespace(name="test.event", payload={"ok": True}))
        for _ in range(3):
            await asyncio.sleep(0)

        legacy_event = json.loads(legacy.sent[0])
        canonical_event = json.loads(canonical.sent[0])
        assert legacy_event["method"] == "axiom.event"
        assert canonical_event["method"] == "zelyr.event"
        assert legacy_event["params"]["event_type"] == "test.event"

    asyncio.run(run())


def test_plugin_manifest_accepts_legacy_version_key(tmp_path) -> None:
    from zelyr.plugins.loader import PluginLoader

    manifest_template = """
[plugin]
name = "legacy-example"
version = "1.2.3"
description = "Legacy plugin"
author = "Legacy Author"
{version_key} = ">=2.0.0"

[entrypoint]
module = "legacy_plugin"
class = "LegacyPlugin"
"""

    for version_key in ("axiom_version", "zelyr_version"):
        plugin_dir = tmp_path / version_key
        plugin_dir.mkdir()
        (plugin_dir / "plugin.toml").write_text(
            manifest_template.format(version_key=version_key),
            encoding="utf-8",
        )
        manifest = PluginLoader().load_manifest(plugin_dir)
        assert manifest.zelyr_version == ">=2.0.0"
        assert manifest.axiom_version == ">=2.0.0"


def test_env_new_name_wins_and_legacy_falls_back(monkeypatch) -> None:
    monkeypatch.delenv("ZELYR_MEMORY_DB", raising=False)
    monkeypatch.delenv("AXIOM_MEMORY_DB", raising=False)

    monkeypatch.setenv("AXIOM_MEMORY_DB", "/tmp/legacy-memory.db")
    assert compat.get_env("MEMORY_DB") == "/tmp/legacy-memory.db"

    monkeypatch.setenv("ZELYR_MEMORY_DB", "/tmp/new-memory.db")
    assert compat.get_env("MEMORY_DB") == "/tmp/new-memory.db"


def test_config_dir_prefers_new_then_existing_legacy(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.delenv("ZELYR_CONFIG_DIR", raising=False)
    monkeypatch.delenv("AXIOM_CONFIG_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: home)

    legacy = home / ".config" / "axiom"
    legacy.mkdir(parents=True)
    assert compat.config_dir() == legacy

    new = home / ".config" / "zelyr"
    new.mkdir(parents=True)
    assert compat.config_dir() == new

    monkeypatch.setenv("AXIOM_CONFIG_DIR", str(tmp_path / "legacy override"))
    assert compat.config_dir() == tmp_path / "legacy override"
    monkeypatch.setenv("ZELYR_CONFIG_DIR", str(tmp_path / "new override"))
    assert compat.config_dir() == tmp_path / "new override"


def test_new_file_uses_existing_legacy_directory(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.delenv("ZELYR_CONFIG_DIR", raising=False)
    monkeypatch.delenv("AXIOM_CONFIG_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: home)

    legacy = home / ".config" / "axiom"
    legacy.mkdir(parents=True)
    assert compat.config_file("instructions.md") == legacy / "instructions.md"

    new = home / ".config" / "zelyr"
    new.mkdir(parents=True)
    legacy_file = legacy / "instructions.md"
    legacy_file.write_text("old", encoding="utf-8")
    assert compat.config_file("instructions.md") == legacy_file

    new_file = new / "instructions.md"
    new_file.write_text("new", encoding="utf-8")
    assert compat.config_file("instructions.md") == new_file


def test_legacy_memory_database_is_reused_without_copy(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ZELYR_MEMORY_DB", raising=False)
    monkeypatch.delenv("AXIOM_MEMORY_DB", raising=False)

    legacy = tmp_path / "axiom_memory.db"
    legacy.write_text("legacy", encoding="utf-8")
    assert compat.repository_memory_db(tmp_path) == legacy
    assert not (tmp_path / "zelyr_memory.db").exists()

    new = tmp_path / "zelyr_memory.db"
    new.write_text("new", encoding="utf-8")
    assert compat.repository_memory_db(tmp_path) == new


def test_legacy_instruction_files_are_read_and_new_names_win(tmp_path) -> None:
    from zelyr.core.instructions import InstructionManager

    legacy = tmp_path / "AXIOM.md"
    legacy.write_text("legacy rules", encoding="utf-8")
    manager = InstructionManager(cwd=tmp_path)
    assert manager.get_project_instructions_path() == legacy
    assert manager.get_project_instructions() == "legacy rules"

    new = tmp_path / "ZELYR.md"
    new.write_text("new rules", encoding="utf-8")
    assert manager.get_project_instructions_path() == new
    assert manager.get_project_instructions() == "new rules"


def test_legacy_theme_id_is_normalized() -> None:
    from zelyr.gui.styles.theme_manager import LEGACY_THEME_IDS, ThemeManager

    assert LEGACY_THEME_IDS["axiom_pro"] == "zelyr_pro"

    class FakeApp:
        stylesheet = ""

        def setStyleSheet(self, qss: str) -> None:
            self.stylesheet = qss

    app = FakeApp()
    manager = ThemeManager()
    manager.apply_theme(app, "axiom_pro")
    assert manager.active_theme_name == "zelyr_pro"
    assert app.stylesheet


def test_packaging_declares_new_and_legacy_entry_points() -> None:
    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)

    scripts = data["project"]["scripts"]
    assert scripts["zelyr"] == scripts["axiom"] == "zelyr.cli.main:main"
    assert scripts["zelyrd"] == scripts["axiomd"] == "zelyr.server.daemon:main"
    assert scripts["zelyr-gui"] == scripts["axiom-gui"] == "zelyr.gui.app:main"
    includes = data["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "zelyr*" in includes
    assert "axiom*" in includes

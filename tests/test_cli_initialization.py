"""Regression tests for ``zelyr.api.cli`` system initialization.

The post-cleanup audit found that ``CLI._register_tools`` imported a module
(``zelyr.tools.vision_tools``) and a class (``VisionCaptureTool``) that no
longer exist.  Because ``_init_system`` is called from ``CLI.__init__``, every
construction of the CLI failed — including the core subsystem of the
``zelyrd`` daemon and the legacy ``main.py`` entry point.

A second startup blocker was found while repairing it: ``zelyr.core.routine``
imports ``croniter`` at module scope and that package is not a declared
dependency, so an optional scheduling backend made the whole CLI unusable.
"""

from __future__ import annotations

import contextlib
import io
import sys

import pytest


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """Construct the real CLI with its home directory isolated to a temp dir."""
    monkeypatch.chdir(tmp_path)
    from zelyr.api.cli import CLI

    instance = CLI()
    try:
        yield instance
    finally:
        instance.close()


def test_cli_initializes_and_registers_replacement_vision_tools(cli):
    """CLI() must complete and expose the vision tools under their real IDs."""
    tools = set(cli.engine.registry.list_tools())

    assert {"echo", "shell", "desktop_control", "screen_capture", "interact_with_ui"} <= tools
    # The removed legacy tool must not resurface as a phantom registration.
    assert "vision_capture" not in tools

    screen_capture = cli.engine.registry.get_tool("screen_capture")
    assert type(screen_capture).__module__ == "zelyr.tools"

    interact_with_ui = cli.engine.registry.get_tool("interact_with_ui")
    assert type(interact_with_ui).__module__ == "zelyr.tools.vision"


def test_cli_starts_when_optional_cron_backend_is_missing(tmp_path, monkeypatch):
    """A missing optional ``croniter`` backend must degrade, not abort startup."""
    monkeypatch.chdir(tmp_path)
    # Force ``import croniter`` to raise ImportError even if it is installed.
    monkeypatch.setitem(sys.modules, "croniter", None)
    monkeypatch.delitem(sys.modules, "zelyr.core.routine", raising=False)

    from zelyr.api.cli import CLI

    instance = CLI()
    try:
        assert instance.routine_engine is None
    finally:
        instance.close()


def _run_cmdloop(cli, stdin_text: str) -> str:
    """Drive ``cmd.Cmd.cmdloop`` with a closed-after-text stdin."""
    cli.stdin = io.StringIO(stdin_text)
    cli.stdout = io.StringIO()
    cli.use_rawinput = False
    # ``cmd.Cmd`` writes prompts to ``self.stdout`` while command handlers use
    # ``print``; capture both streams to assert on the full CLI session.
    with contextlib.redirect_stdout(cli.stdout):
        cli.cmdloop()
    return cli.stdout.getvalue()


def test_cmdloop_eof_terminates_cleanly(cli):
    """Closed stdin must exit instead of spinning on the ``EOF`` sentinel."""
    output = _run_cmdloop(cli, "")

    assert cli._closed is True
    assert "Shutting down Zelyr..." in output
    assert "Goodbye!" in output


def test_cmdloop_slash_exit_still_terminates(cli):
    """The documented ``/exit`` command must keep working through ``precmd``."""
    output = _run_cmdloop(cli, "/exit\n")

    assert cli._closed is True
    assert "Goodbye!" in output


def test_cmdloop_bare_exit_does_not_reach_the_model(cli, monkeypatch):
    """A bare ``exit`` is a command, not a chat prompt sent as ``ask exit``."""
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("bare exit was routed to do_ask")

    monkeypatch.setattr(cli, "do_ask", fail_if_called)
    output = _run_cmdloop(cli, "exit\n")

    assert cli._closed is True
    assert "Goodbye!" in output


def test_cmdloop_ordinary_command_runs_then_eof_exits(cli):
    """Normal commands execute and are followed by a clean EOF shutdown."""
    output = _run_cmdloop(cli, "/help\n")

    assert "Zelyr Commands" in output
    assert "Goodbye!" in output

"""Focused tests for the isolated distribution build helper."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts import build_dist


def test_copy_inputs_includes_packaging_files_and_excludes_caches(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    package = root / "zelyr" / "pkg"
    package.mkdir(parents=True)
    (root / "zelyr" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "module.py").write_text("VALUE = 1\n")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-311.pyc").write_bytes(b"stale")
    (root / "pyproject.toml").write_text("[build-system]\n")
    (root / "README.md").write_text("readme\n")

    monkeypatch.setattr(build_dist, "ROOT", root)
    destination = tmp_path / "destination"
    destination.mkdir()
    build_dist._copy_inputs(destination)

    assert (destination / "pyproject.toml").is_file()
    assert (destination / "README.md").is_file()
    assert (destination / "zelyr" / "pkg" / "module.py").is_file()
    assert not (destination / "zelyr" / "pkg" / "__pycache__").exists()


def _write_wheel(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)


def test_verify_wheel_rejects_module_absent_from_source(tmp_path):
    source = tmp_path / "source"
    (source / "zelyr").mkdir(parents=True)
    (source / "zelyr" / "present.py").write_text("")
    wheel = tmp_path / "zelyr_core-1.0.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        {
            "zelyr/present.py": "",
            "zelyr/deleted_module.py": "",
        },
    )

    with pytest.raises(SystemExit, match="zelyr/deleted_module.py"):
        build_dist._verify_wheel(wheel, source)


def test_verify_wheel_accepts_modules_present_in_source(tmp_path):
    source = tmp_path / "source"
    (source / "zelyr").mkdir(parents=True)
    (source / "zelyr" / "present.py").write_text("")
    wheel = tmp_path / "zelyr_core-1.0.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        {
            "zelyr/present.py": "",
            "zelyr_core-1.0.0.dist-info/METADATA": "Name: zelyr-core\n",
        },
    )

    build_dist._verify_wheel(wheel, source)


def test_main_cleans_temp_source_on_failure_and_preserves_outdir(
    tmp_path, monkeypatch
):
    outdir = tmp_path / "dist"
    outdir.mkdir()
    sentinel = outdir / "keep.txt"
    sentinel.write_text("keep me")
    temp_root = tmp_path / "zelyr-build-src-test"
    temp_root.mkdir()

    monkeypatch.setattr(build_dist.tempfile, "mkdtemp", lambda **_: str(temp_root))
    monkeypatch.setattr(
        build_dist, "_copy_inputs", lambda destination: destination.mkdir(exist_ok=True)
    )

    def fail_build(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(build_dist.subprocess, "run", fail_build)
    monkeypatch.setattr(
        sys, "argv", ["build_dist.py", "--wheel", "--outdir", str(outdir)]
    )

    with pytest.raises(subprocess.CalledProcessError):
        build_dist.main()

    assert not temp_root.exists()
    assert sentinel.read_text() == "keep me"
    assert outdir.is_dir()


def test_main_builds_from_isolated_source_and_cleans_up(tmp_path, monkeypatch):
    outdir = tmp_path / "dist"
    temp_root = tmp_path / "zelyr-build-src-success"
    temp_root.mkdir()
    wheel = outdir / "zelyr_core-1.0.0-py3-none-any.whl"
    observed: dict[str, object] = {}

    monkeypatch.setattr(build_dist.tempfile, "mkdtemp", lambda **_: str(temp_root))
    monkeypatch.setattr(
        build_dist, "_copy_inputs", lambda destination: destination.mkdir(exist_ok=True)
    )
    monkeypatch.setattr(build_dist, "_verify_wheel", lambda *_args: None)

    def fake_build(command, cwd, check):
        observed["command"] = command
        observed["cwd"] = Path(cwd)
        outdir.mkdir(parents=True, exist_ok=True)
        wheel.write_bytes(b"wheel")

    monkeypatch.setattr(build_dist.subprocess, "run", fake_build)
    monkeypatch.setattr(sys, "argv", ["build_dist.py", "--wheel", "--outdir", str(outdir)])

    assert build_dist.main() == 0
    assert observed["cwd"] == temp_root / "src"
    assert wheel.is_file()
    assert not temp_root.exists()

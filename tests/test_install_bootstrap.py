"""Isolated preflight test suite for reproducible installation and runtime bootstrap."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


class TestInstallBootstrap(unittest.TestCase):
    """Test suite covering first-run directory initialization, path portability, and packaging."""

    def test_first_run_directory_auto_bootstrap_isolated(self) -> None:
        """Verify that launching Zelyr when ~/.config/zelyr does not exist creates all required directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_home = Path(tmp_dir) / "home"
            temp_home.mkdir(parents=True, exist_ok=True)
            custom_config = temp_home / ".config" / "zelyr"

            self.assertFalse(custom_config.exists())

            # Test initialize_directories with custom isolated base path
            from zelyr.config import initialize_directories, get_config_dir

            initialize_directories(config_dir=custom_config)

            self.assertTrue(custom_config.exists())
            self.assertTrue((custom_config / "tools.d").is_dir())
            self.assertTrue((custom_config / "plugins").is_dir())
            self.assertTrue((custom_config / "sessions").is_dir())
            self.assertTrue((custom_config / "logs").is_dir())
            self.assertTrue((custom_config / "models").is_dir())

    def test_environment_override_config_dir(self) -> None:
        """Verify that ZELYR_CONFIG_DIR environment variable correctly overrides the config directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            override_path = Path(tmp_dir) / "custom_zelyr_conf"
            old_env = os.environ.get("ZELYR_CONFIG_DIR")
            try:
                os.environ["ZELYR_CONFIG_DIR"] = str(override_path)
                from zelyr.config import get_config_dir, initialize_directories

                resolved_dir = get_config_dir()
                self.assertEqual(resolved_dir.resolve(), override_path.resolve())

                initialize_directories(resolved_dir)
                self.assertTrue(override_path.exists())
                self.assertTrue((override_path / "tools.d").is_dir())
            finally:
                if old_env is not None:
                    os.environ["ZELYR_CONFIG_DIR"] = old_env
                else:
                    os.environ.pop("ZELYR_CONFIG_DIR", None)

    def _get_repo_root(self) -> Path:
        current = Path(__file__).resolve().parent
        if (current / "pyproject.toml").is_file():
            return current
        return current.parent

    def test_zero_hardcoded_user_paths_in_zelyr(self) -> None:
        """Audit the entire zelyr/ source tree to ensure zero hardcoded personal user home paths exist."""
        repo_root = self._get_repo_root()
        zelyr_pkg = repo_root / "zelyr"

        violations = []
        for py_path in zelyr_pkg.rglob("*.py"):
            content = py_path.read_text(encoding="utf-8")
            # Disallow specific host user names and absolute /home/<specific_user> paths
            if "/home/rutytoi" in content:
                violations.append(f"{py_path.relative_to(repo_root)} contains '/home/rutytoi'")
            if "/var/home/rutytoi" in content:
                violations.append(f"{py_path.relative_to(repo_root)} contains '/var/home/rutytoi'")

        self.assertEqual(violations, [], f"Hardcoded developer paths found in production code:\n" + "\n".join(violations))

    def test_pyproject_entrypoints_and_callables(self) -> None:
        """Verify pyproject.toml scripts point to valid, callable entry points."""
        repo_root = self._get_repo_root()
        pyproject_path = repo_root / "pyproject.toml"
        self.assertTrue(pyproject_path.is_file(), "pyproject.toml must exist")

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        project = data.get("project", {})
        scripts = project.get("scripts", {})
        self.assertIn("zelyr", scripts, "pyproject.toml must define an 'zelyr' console script entry point")
        self.assertIn("zelyrd", scripts, "pyproject.toml must define an 'zelyrd' console script entry point")

        for script_name, entry_point in scripts.items():
            self.assertIn(":", entry_point, f"Entry point for '{script_name}' must be 'module:callable'")
            mod_name, func_name = entry_point.split(":", 1)
            mod = importlib.import_module(mod_name)
            self.assertTrue(hasattr(mod, func_name), f"Module '{mod_name}' must export '{func_name}' for '{script_name}'")
            func = getattr(mod, func_name)
            self.assertTrue(callable(func), f"'{entry_point}' for '{script_name}' must be callable")

    def test_pyproject_dependencies_coverage(self) -> None:
        """Verify essential runtime dependencies are declared in pyproject.toml."""
        repo_root = self._get_repo_root()
        pyproject_path = repo_root / "pyproject.toml"

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        deps = data.get("project", {}) .get("dependencies", [])
        dep_names = {d.split(">=")[0].split("==")[0].split("<")[0].split("[")[0].strip().lower() for d in deps}

        required_core = {"fastapi", "httpx", "prompt_toolkit", "requests", "websockets", "rich", "pydantic"}
        missing = required_core - dep_names
        self.assertEqual(missing, set(), f"Essential runtime dependencies missing from pyproject.toml: {missing}")


if __name__ == "__main__":
    unittest.main()

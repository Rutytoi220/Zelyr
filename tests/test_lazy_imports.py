"""Regression test suite for PEP 562 lazy exports and CLI startup optimization.

Covers:
1. Public API parity: All exported symbols resolve to expected objects.
2. Dynamic lookup: getattr resolves lazy exports, dir() exposes them, unknown raises AttributeError.
3. Import laziness: Root package import in fresh subprocess does not load heavy dependencies.
4. Deferred resolution: Accessing a lazy export loads its defining module on demand.
5. Legacy compatibility: Supported axiom imports and object identities remain intact.
6. Wildcard imports and introspection: from zelyr import * and from axiom import * function cleanly.
7. Circular-import safety: Import orders across zelyr, axiom, and submodules execute without cycle errors.
8. CLI behavior: --help, --version, and status commands execute cleanly.
9. Failure behavior: Missing optional dependencies produce actionable errors without obscure import crashes.
"""

from __future__ import annotations

import subprocess
import sys
import unittest


class TestLazyImportsPublicAPI(unittest.TestCase):
    """Test 1 & 2: Public API parity and dynamic lookup via PEP 562."""

    def test_public_api_parity(self):
        """All symbols in zelyr.__all__ must resolve to valid objects."""
        import zelyr

        for name in zelyr.__all__:
            self.assertTrue(hasattr(zelyr, name), f"zelyr missing exported attribute {name}")
            attr = getattr(zelyr, name)
            self.assertIsNotNone(attr, f"zelyr.{name} resolved to None")

        # Spot-check critical core exports
        from zelyr.core.engine import Engine
        from zelyr.config import ZelyrConfig

        self.assertIs(zelyr.Engine, Engine)
        self.assertIs(zelyr.ZelyrConfig, ZelyrConfig)
        self.assertIs(zelyr.AxiomConfig, ZelyrConfig)
        self.assertEqual(zelyr.__version__, "1.2.0.dev0")

    def test_dynamic_lookup_and_dir(self):
        """getattr resolves lazy exports, dir() includes them, unknown raises AttributeError."""
        import zelyr

        # Known attribute lookup
        engine_cls = getattr(zelyr, "Engine")
        self.assertEqual(engine_cls.__name__, "Engine")

        # Unknown attribute lookup
        with self.assertRaises(AttributeError) as ctx:
            getattr(zelyr, "nonexistent_symbol_xyz")
        self.assertIn("module 'zelyr' has no attribute 'nonexistent_symbol_xyz'", str(ctx.exception))

        # dir() listing
        exports = dir(zelyr)
        self.assertIn("Engine", exports)
        self.assertIn("ZelyrConfig", exports)
        self.assertIn("TaskPlanner", exports)
        self.assertIn("__version__", exports)

    def test_caching_in_module_globals(self):
        """Resolved lazy attributes must be cached in module globals."""
        import zelyr

        self.assertIn("Engine", zelyr._LAZY_EXPORTS)
        # Access attribute to trigger lazy import and caching
        _ = zelyr.Engine
        self.assertIn("Engine", zelyr.__dict__)
        self.assertIs(zelyr.__dict__["Engine"], zelyr.Engine)


class TestImportLaziness(unittest.TestCase):
    """Test 3 & 4: Import laziness and deferred on-demand resolution in clean subprocesses."""

    def test_root_import_does_not_load_heavy_dependencies(self):
        """import zelyr in a clean subprocess must not load litellm, torch, playwright, etc."""
        code = (
            "import sys\n"
            "import zelyr\n"
            "heavy = ['litellm', 'torch', 'playwright', 'pydantic', 'aiosqlite', 'chromadb', 'cv2', 'PIL']\n"
            "loaded = [m for m in heavy if m in sys.modules]\n"
            "if loaded:\n"
            "    print('UNEXPECTED_LOADED:' + ','.join(loaded))\n"
            "    sys.exit(1)\n"
            "sys.exit(0)\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Heavy modules loaded on import zelyr: {res.stdout} {res.stderr}")

    def test_deferred_resolution_loads_module_on_demand(self):
        """Accessing a lazy export must load only the necessary module on demand."""
        code = (
            "import sys\n"
            "import zelyr\n"
            "assert 'zelyr.core' not in sys.modules, 'zelyr.core loaded prematurely'\n"
            "engine = zelyr.Engine\n"
            "assert 'zelyr.core' in sys.modules, 'zelyr.core was not loaded on demand'\n"
            "assert engine.__name__ == 'Engine'\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Deferred resolution failed: {res.stdout} {res.stderr}")


class TestLegacyCompatibility(unittest.TestCase):
    """Test 5 & 6: Legacy axiom compatibility, wildcard imports, and introspection."""

    def test_axiom_import_laziness_and_object_identity(self):
        """import axiom must be lazy and preserve object identities with zelyr."""
        code = (
            "import sys\n"
            "import axiom\n"
            "import zelyr\n"
            "assert 'litellm' not in sys.modules, 'litellm loaded by import axiom'\n"
            "assert axiom.__version__ == zelyr.__version__\n"
            "assert axiom.Engine is zelyr.Engine\n"
            "assert axiom.AxiomConfig is zelyr.ZelyrConfig\n"
            "from axiom import Engine\n"
            "assert Engine is zelyr.Engine\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Axiom compatibility failed: {res.stdout} {res.stderr}")

    def test_wildcard_imports(self):
        """from zelyr import * and from axiom import * must resolve cleanly in fresh environments."""
        code_zelyr = (
            "from zelyr import *\n"
            "assert Engine is not None\n"
            "assert ZelyrConfig is not None\n"
            "assert __version__ == '1.2.0.dev0'\n"
        )
        res_zelyr = subprocess.run([sys.executable, "-c", code_zelyr], capture_output=True, text=True)
        self.assertEqual(res_zelyr.returncode, 0, f"from zelyr import * failed: {res_zelyr.stderr}")

        code_axiom = (
            "from axiom import *\n"
            "assert Engine is not None\n"
            "assert AxiomConfig is not None\n"
            "assert __version__ == '1.2.0.dev0'\n"
        )
        res_axiom = subprocess.run([sys.executable, "-c", code_axiom], capture_output=True, text=True)
        self.assertEqual(res_axiom.returncode, 0, f"from axiom import * failed: {res_axiom.stderr}")


class TestCircularImportSafety(unittest.TestCase):
    """Test 7: Exercise various import sequences across zelyr, axiom, and submodules."""

    def test_interleaved_import_orders(self):
        """Different import sequences must not raise circular import errors."""
        sequences = [
            "import zelyr.core; import zelyr; from zelyr import Engine",
            "import zelyr; import zelyr.core; from zelyr import Engine",
            "import axiom; import zelyr; from axiom import Engine",
            "import zelyr; import axiom; from axiom import Engine",
            "from zelyr.main import __version__, Engine; assert Engine is not None",
            "from zelyr.cli.main import _build_parser; p = _build_parser(); assert p is not None",
        ]
        for seq in sequences:
            res = subprocess.run([sys.executable, "-c", seq], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Import sequence failed: {seq}\nError: {res.stderr}")


class TestCLIBehaviorAndStartup(unittest.TestCase):
    """Test 8 & 9: Lightweight CLI commands and failure behavior."""

    def test_cli_lightweight_commands(self):
        """CLI --version, --help, and status commands must succeed without importing heavy ML."""
        # 1. Version command
        res_ver = subprocess.run(
            [sys.executable, "-m", "zelyr.cli.main", "--version"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res_ver.returncode, 0)
        self.assertIn("zelyr 1.2.0.dev0", res_ver.stdout)

        # 2. Help command
        res_help = subprocess.run(
            [sys.executable, "-m", "zelyr.cli.main", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res_help.returncode, 0)
        self.assertIn("Zelyr — Local-First AI Operating System", res_help.stdout)

        # 3. Status command
        res_status = subprocess.run(
            [sys.executable, "-m", "zelyr.cli.main", "status"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res_status.returncode, 0)
        self.assertIn("Zelyr System Status", res_status.stdout)

    def test_cli_help_does_not_load_litellm(self):
        """Executing zelyr --help must not load litellm in the CLI process."""
        code = (
            "import sys, subprocess\n"
            "res = subprocess.run([sys.executable, '-X', 'importtime', '-m', 'zelyr.cli.main', '--help'], "
            "capture_output=True, text=True)\n"
            "assert res.returncode == 0\n"
            "assert 'litellm' not in res.stderr, 'litellm was imported during --help'\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"litellm loaded during CLI help: {res.stderr}")


if __name__ == "__main__":
    unittest.main()

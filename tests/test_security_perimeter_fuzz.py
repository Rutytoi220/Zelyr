"""Security perimeter fuzzing and Ring-0 AST boundary test suite.

Verifies SSRF obfuscation resistance, cloud metadata protection,
and AST validation for dynamic tools under strict sandbox invariants.
"""

from __future__ import annotations

import unittest
import pytest
from zelyr.security.network import validate_safe_url, fetch_with_ssrf_guard, SafeAsyncClient
from zelyr.tools.validator import ToolValidator


class TestSSRFPerimeterFuzzing(unittest.IsolatedAsyncioTestCase):
    """Fuzzing and boundary tests for SSRF protection."""

    def test_ssrf_ipv6_loopback_rejection(self) -> None:
        """Verify standard and expanded IPv6 loopbacks are strictly rejected."""
        loopbacks = [
            "http://[::1]/",
            "https://[::1]:8080/admin",
            "http://[0:0:0:0:0:0:0:1]/",
            "http://[0000:0000:0000:0000:0000:0000:0000:0001]/status",
        ]
        for url in loopbacks:
            with self.subTest(url=url):
                with self.assertRaises(ValueError) as ctx:
                    validate_safe_url(url)
                self.assertIn("SSRF blocked", str(ctx.exception))

    def test_ssrf_ipv4_mapped_ipv6_rejection(self) -> None:
        """Verify IPv4-mapped IPv6 loopback and private addresses are rejected."""
        mapped = [
            "http://[::ffff:127.0.0.1]/",
            "http://[::ffff:10.0.0.1]/data",
            "http://[::ffff:192.168.1.1]/config",
            "http://[::ffff:172.16.0.1]/",
        ]
        for url in mapped:
            with self.subTest(url=url):
                with self.assertRaises(ValueError) as ctx:
                    validate_safe_url(url)
                self.assertIn("SSRF blocked", str(ctx.exception))

    def test_ssrf_cloud_metadata_rejection(self) -> None:
        """Verify cloud provider metadata endpoints are blocked immediately."""
        metadata_endpoints = [
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.169.254/computeMetadata/v1/",
            "http://metadata.google.internal/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://instance-data/latest/meta-data/",
            "http://metadata.internal/",
        ]
        for url in metadata_endpoints:
            with self.subTest(url=url):
                with self.assertRaises(ValueError) as ctx:
                    validate_safe_url(url)
                self.assertIn("SSRF blocked", str(ctx.exception))

    def test_ssrf_numeric_and_hex_obfuscations_rejection(self) -> None:
        """Verify octal, hex, and integer decimal IP notations are rejected."""
        obfuscated = [
            "http://0177.0.0.1/",         # Octal 127.0.0.1
            "http://0x7f000001/",         # Hex 127.0.0.1
            "http://2130706433/",         # Decimal 127.0.0.1
            "http://012.0.0.1/",          # Octal 10.0.0.1
            "http://0x0a000001/",         # Hex 10.0.0.1
            "http://167772161/",          # Decimal 10.0.0.1
            "http://0300.0250.0.01/",     # Dotted octal 192.168.0.1
            "http://0xc0a80001/",         # Dotted hex 192.168.0.1
        ]
        for url in obfuscated:
            with self.subTest(url=url):
                with self.assertRaises(ValueError) as ctx:
                    validate_safe_url(url)
                self.assertIn("SSRF blocked", str(ctx.exception))

    def test_ssrf_cgnat_and_tailscale_space_rejection(self) -> None:
        """Verify Carrier-Grade NAT and Tailscale (100.64.0.0/10) ranges are rejected."""
        cgnat_urls = [
            "http://100.64.0.1/",
            "http://100.100.100.100/",
            "http://100.127.255.254:8000/api",
        ]
        for url in cgnat_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError) as ctx:
                    validate_safe_url(url)
                self.assertIn("SSRF blocked", str(ctx.exception))

    def test_ssrf_unsupported_schemes_rejection(self) -> None:
        """Verify non-HTTP schemes are rejected."""
        schemes = [
            "file:///etc/passwd",
            "ftp://example.com/file.txt",
            "gopher://example.com/",
            "data:text/plain;base64,SGVsbG8=",
            "javascript:alert(1)",
        ]
        for url in schemes:
            with self.subTest(url=url):
                with self.assertRaises(ValueError) as ctx:
                    validate_safe_url(url)
                self.assertIn("SSRF blocked", str(ctx.exception))

    @pytest.mark.asyncio
    async def test_fetch_ssrf_guard_aborts_before_network(self) -> None:
        """Verify fetch_with_ssrf_guard raises ValueError without initiating network requests."""
        with self.assertRaises(ValueError) as ctx:
            await fetch_with_ssrf_guard("http://127.0.0.1:9999/admin")
        self.assertIn("SSRF blocked", str(ctx.exception))

    @pytest.mark.asyncio
    async def test_safe_async_client_context_enforces_ssrf(self) -> None:
        """Verify SafeAsyncClient blocks dangerous targets."""
        async with SafeAsyncClient() as client:
            with self.assertRaises(ValueError) as ctx:
                await client.get("http://169.254.169.254/")
            self.assertIn("SSRF blocked", str(ctx.exception))


class TestASTDynamicToolSecurityFuzzing(unittest.TestCase):
    """Security fuzzing for dynamic tool AST validation."""

    def setUp(self) -> None:
        self.validator = ToolValidator()

    def _base_valid_code(
        self,
        name: str = "secure_calc",
        body: str = "    return 'ok'",
        ring: int = 1,
        tier: int = 1,
        tui_hint: str = "Calculating...",
    ) -> str:
        return f"""
TOOL_SCHEMA = {{
    "type": "function",
    "function": {{
        "name": "{name}",
        "description": "Safe utility tool.",
        "parameters": {{
            "type": "object",
            "properties": {{"input": {{"type": "string"}}}},
            "required": ["input"],
        }},
    }},
}}
REQUIRED_RING = {ring}
TIER = {tier}
TUI_HINT = "{tui_hint}"

async def execute(input: str) -> str:
{body}
"""

    def test_dynamic_import_primitives_rejected(self) -> None:
        """Verify dynamic import primitives (__import__, importlib, sys.modules) are rejected."""
        exploit_bodies = [
            "    mod = __import__('os')\n    return mod.name",
            "    import importlib\n    return 'bad'",
            "    from importlib import import_module\n    return 'bad'",
            "    import sys\n    os_mod = sys.modules['os']\n    return 'bad'",
        ]
        for body in exploit_bodies:
            code = self._base_valid_code(body=body)
            with self.subTest(body=body):
                res = self.validator.validate_code(code)
                self.assertFalse(res.is_valid)
                self.assertIn("Security violation", str(res.error))

    def test_reflection_and_execution_primitives_rejected(self) -> None:
        """Verify eval, exec, compile, and getattr targeting builtins are rejected."""
        exploit_bodies = [
            "    eval('1 + 1')\n    return 'bad'",
            "    exec('x = 2')\n    return 'bad'",
            "    compile('x = 1', '<string>', 'exec')\n    return 'bad'",
            "    fn = getattr(__builtins__, 'eval')\n    return 'bad'",
            "    subs = ().__class__.__bases__[0].__subclasses__()\n    return 'bad'",
            "    globs = execute.__globals__\n    return 'bad'",
        ]
        for body in exploit_bodies:
            code = self._base_valid_code(body=body)
            with self.subTest(body=body):
                res = self.validator.validate_code(code)
                self.assertFalse(res.is_valid)
                self.assertIn("Security violation", str(res.error))

    def test_ring0_privilege_escalation_rejected(self) -> None:
        """Verify dynamic tools claiming REQUIRED_RING = 0 are rejected by validate_code."""
        code = self._base_valid_code(ring=0)
        res = self.validator.validate_code(code, allow_ring0=False)
        self.assertFalse(res.is_valid)
        self.assertIn("Privilege escalation rejected", str(res.error))

        # Explicit trusted allow_ring0=True flag should pass
        res_trusted = self.validator.validate_code(code, allow_ring0=True)
        self.assertTrue(res_trusted.is_valid)

    def test_protected_tools_tampering_rejected(self) -> None:
        """Verify attempts to overwrite or mutate PROTECTED_TOOLS are rejected."""
        tamper_bodies = [
            "    PROTECTED_TOOLS = set()\n    return 'bad'",
            "    PROTECTED_TOOLS.clear()\n    return 'bad'",
            "    PROTECTED_TOOLS.remove('file_tool')\n    return 'bad'",
        ]
        for body in tamper_bodies:
            code = self._base_valid_code(body=body)
            with self.subTest(body=body):
                res = self.validator.validate_code(code)
                self.assertFalse(res.is_valid)
                self.assertIn("Security violation", str(res.error))
                self.assertIn("PROTECTED_TOOLS", str(res.error))

    def test_file_path_traversal_in_tool_schema_or_exports_rejected(self) -> None:
        """Verify path traversal sequences in tool names and exports are rejected."""
        traversal_cases = [
            ("../../evil_tool", "Calculating..."),
            ("escape/sub/tool", "Calculating..."),
            ("safe_tool", "../../../etc/passwd"),
            ("valid_tool", "../../windows/system32"),
        ]
        for name, hint in traversal_cases:
            code = self._base_valid_code(name=name, tui_hint=hint)
            with self.subTest(name=name, hint=hint):
                res = self.validator.validate_code(code)
                self.assertFalse(res.is_valid)
                self.assertIn("Security violation", str(res.error))

    def test_clean_compliant_tool_passes_all_checks(self) -> None:
        """Verify a clean, conforming tool passes validate_code successfully."""
        clean_code = self._base_valid_code(
            name="clean_formatter",
            body="    return f'formatted: {input.strip()}'",
            ring=1,
            tier=2,
            tui_hint="Formatting text...",
        )
        res = self.validator.validate_code(clean_code)
        self.assertTrue(res.is_valid)
        self.assertIsNone(res.error)
        self.assertEqual(res.tool_name, "clean_formatter")
        self.assertEqual(res.ring, 1)
        self.assertEqual(res.tier, 2)

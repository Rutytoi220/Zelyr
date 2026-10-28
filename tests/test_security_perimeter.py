"""Security perimeter regression tests for AST sandboxing, SSRF defenses, and tool authorization."""

from __future__ import annotations

import socket
import unittest.mock
import pytest
import httpx

from zelyr.security.ast_validator import SecurityASTValidator, SecurityViolationError
from zelyr.security.network import (
    PinnedIPNetworkBackend,
    PinnedIPTransport,
    fetch_with_ssrf_guard,
    validate_safe_url,
)
from zelyr.tool_registry import ToolRegistry
from zelyr.tools import PythonExecTool


def _mock_public_dns(host, port=None, *args, **kwargs):
    """Deterministic DNS resolver returning public IP for benign domains and loopback for localhost."""
    clean_host = str(host).strip("[]").strip().rstrip(".").lower()
    if clean_host in ("localhost", "127.0.0.1"):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]


class TestASTSandboxing:
    """Verifies that dynamic Python execution blocks dangerous AST nodes and executes safe code."""

    @pytest.mark.parametrize(
        "payload,expected_rule",
        [
            ("import ctypes", "FORBIDDEN_IMPORT"),
            ("from os import system", "FORBIDDEN_IMPORT"),
            ("import subprocess", "FORBIDDEN_IMPORT"),
            ("import sys", "FORBIDDEN_IMPORT"),
            ("import shutil", "FORBIDDEN_IMPORT"),
            ("import importlib", "FORBIDDEN_IMPORT"),
            ("from builtins import exec", "FORBIDDEN_IMPORT"),
        ],
    )
    def test_forbidden_imports(self, payload: str, expected_rule: str) -> None:
        """Assert dangerous imports raise SecurityViolationError and are rejected by PythonExecTool."""
        with pytest.raises(SecurityViolationError) as exc_info:
            SecurityASTValidator.validate(payload)
        assert exc_info.value.rule_id == expected_rule

        tool = PythonExecTool()
        res = tool.execute(code=payload)
        assert res.success is False
        assert "SecurityViolation" in str(res.error)
        assert expected_rule in str(res.error)

    @pytest.mark.parametrize(
        "payload",
        [
            "().__class__.__base__.__subclasses__()",
            "[].__class__.__mro__",
            "f = lambda: None; f.__globals__",
            "f = lambda: None; f.__code__",
            "x = object().__class__.__dict__",
        ],
    )
    def test_reflection_and_dunder_escapes(self, payload: str) -> None:
        """Assert dunder reflection and traversal are blocked by AST inspection."""
        with pytest.raises(SecurityViolationError):
            SecurityASTValidator.validate(payload)

        tool = PythonExecTool()
        res = tool.execute(code=payload)
        assert res.success is False
        assert "SecurityViolation" in str(res.error)

    @pytest.mark.parametrize(
        "payload",
        [
            "getattr(__builtins__, '__import__')",
            "eval('1 + 1')",
            "exec('pass')",
            "compile('pass', '', 'exec')",
            "open('/etc/passwd', 'r')",
            "breakpoint()",
        ],
    )
    def test_dynamic_builtins_and_getattr(self, payload: str) -> None:
        """Assert dangerous built-in invocations and dynamic lookups are blocked."""
        with pytest.raises(SecurityViolationError):
            SecurityASTValidator.validate(payload)

        tool = PythonExecTool()
        res = tool.execute(code=payload)
        assert res.success is False
        assert "SecurityViolation" in str(res.error)

    def test_legitimate_computational_python(self) -> None:
        """Assert legitimate mathematical, algorithmic, and string formatting code runs cleanly."""
        code = (
            "import math\n"
            "def fib(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
            "res = fib(10)\n"
            "sq = math.sqrt(144)\n"
            "print(f'RESULT: sqrt={sq}, fib={res}')\n"
        )
        SecurityASTValidator.validate(code)

        tool = PythonExecTool()
        res = tool.execute(code=code)
        assert res.success is True
        assert res.output is not None
        assert "RESULT: sqrt=12.0, fib=55" in res.output["stdout"]

    def test_syntax_error_handling(self) -> None:
        """Assert invalid Python syntax returns a structured SyntaxError ToolResult."""
        code = "def broken_func("
        tool = PythonExecTool()
        res = tool.execute(code=code)
        assert res.success is False
        assert "SyntaxError" in str(res.error)


class TestSSRFProtection:
    """Verifies strict SSRF guards, DNS classification, IP pinning, and redirect defenses."""

    def test_reject_localhost(self) -> None:
        """Assert localhost hostnames and loopback IP literals are rejected."""
        with pytest.raises(ValueError, match="SSRF blocked"):
            validate_safe_url("http://127.0.0.1:8000")

        with pytest.raises(ValueError, match="SSRF blocked"):
            validate_safe_url("http://localhost")

        with pytest.raises(ValueError, match="SSRF blocked"):
            validate_safe_url("http://127.0.0.1:11434")

    @pytest.mark.parametrize(
        "url",
        [
            "http://10.0.0.1/api",
            "http://192.168.1.1:8080",
            "http://172.16.0.1/admin",
            "http://172.31.255.255/status",
        ],
    )
    def test_reject_private_ranges(self, url: str) -> None:
        """Assert RFC 1918 private IP subnets are strictly rejected."""
        with pytest.raises(ValueError, match="SSRF blocked"):
            validate_safe_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data",
            "http://metadata.google.internal/computeMetadata/v1",
            "http://instance-data/latest/meta-data",
        ],
    )
    def test_reject_cloud_metadata(self, url: str) -> None:
        """Assert cloud metadata IP addresses and hostnames are rejected."""
        with pytest.raises(ValueError, match="SSRF blocked"):
            validate_safe_url(url)

    def test_reject_ipv6_loopback_and_mapped(self) -> None:
        """Assert IPv6 loopback and IPv4-mapped loopback ranges are rejected."""
        with pytest.raises(ValueError, match="SSRF blocked"):
            validate_safe_url("http://[::1]:80")

        with pytest.raises(ValueError, match="SSRF blocked"):
            validate_safe_url("http://[::ffff:127.0.0.1]:8080")

    def test_reject_embedded_credentials(self) -> None:
        """Assert URLs containing embedded userinfo credentials are rejected."""
        with pytest.raises(ValueError, match="Embedded credentials in URL are disallowed"):
            validate_safe_url("http://admin:secret@example.com/data")

    def test_legitimate_public_url_passes(self) -> None:
        """Assert public domain name with safe IP passes validation."""
        with unittest.mock.patch("socket.getaddrinfo", side_effect=_mock_public_dns):
            validated = validate_safe_url("https://example.com/safe/resource")
            assert validated == "https://example.com/safe/resource"

    @pytest.mark.asyncio
    async def test_reject_redirect_to_private_ip(self) -> None:
        """Assert intermediate 3xx redirect hopping to private subnet is aborted."""
        redirect_resp = httpx.Response(
            status_code=302,
            headers={"Location": "http://192.168.1.1/secret"},
            request=httpx.Request("GET", "https://example.com/login"),
        )
        mock_client = unittest.mock.AsyncMock(spec=httpx.AsyncClient)
        mock_client.request = unittest.mock.AsyncMock(return_value=redirect_resp)

        with unittest.mock.patch("socket.getaddrinfo", side_effect=_mock_public_dns):
            with pytest.raises(ValueError, match="SSRF blocked"):
                await fetch_with_ssrf_guard("https://example.com/login", client=mock_client)

    @pytest.mark.asyncio
    async def test_dns_rebinding_simulation(self) -> None:
        """Simulate DNS rebinding attack where hostname resolves to public IP initially and private IP later."""
        dns_call_count = 0

        def _rebinding_dns(host, port=None, *args, **kwargs):
            nonlocal dns_call_count
            dns_call_count += 1
            if dns_call_count == 1:
                # Pre-flight lookup returns legitimate public IP
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
            else:
                # Rebound lookup attempts loopback redirect
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]

        with unittest.mock.patch("socket.getaddrinfo", side_effect=_rebinding_dns):
            # Preflight validation passes on turn 1
            validated = validate_safe_url("http://rebind-target.com/page")
            assert validated == "http://rebind-target.com/page"

            # Connect-time resolution occurs on turn 2 via PinnedIPNetworkBackend
            backend = PinnedIPNetworkBackend()
            with pytest.raises(ValueError, match="SSRF blocked"):
                await backend.connect_tcp("rebind-target.com", 80)

    @pytest.mark.asyncio
    async def test_max_redirects_exceeded(self) -> None:
        """Assert that redirects exceeding max_redirects threshold are aborted."""
        redirect_resp = httpx.Response(
            status_code=302,
            headers={"Location": "https://example.com/next"},
            request=httpx.Request("GET", "https://example.com/next"),
        )
        mock_client = unittest.mock.AsyncMock(spec=httpx.AsyncClient)
        mock_client.request = unittest.mock.AsyncMock(return_value=redirect_resp)

        with unittest.mock.patch("socket.getaddrinfo", side_effect=_mock_public_dns):
            with pytest.raises(ValueError, match="Too many redirects"):
                await fetch_with_ssrf_guard(
                    "https://example.com/start",
                    client=mock_client,
                    max_redirects=2,
                )


class TestToolCapabilityAuthorization:
    """Verifies caller ring boundaries and allowed tool list enforcement in ToolRegistry."""

    @pytest.mark.asyncio
    async def test_unauthorized_ring_call_rejected(self) -> None:
        """Assert unprivileged caller (Ring 2) attempting to invoke Ring 0 tool is rejected."""
        registry = ToolRegistry()
        res = await registry.execute_async(
            "shell",
            caller_ring=2,
            command="whoami",
        )
        assert res.success is False
        assert "AuthorizationError" in str(res.error)
        assert "requires higher permissions" in str(res.error)

    @pytest.mark.asyncio
    async def test_authorized_ring_call_succeeds(self) -> None:
        """Assert privileged caller (Ring 0) executing Ring 0 tool succeeds."""
        registry = ToolRegistry()
        res = await registry.execute_async(
            "echo",
            caller_ring=0,
            text="hello security",
        )
        assert res.success is True
        assert res.output == "hello security"

    @pytest.mark.asyncio
    async def test_allowed_tools_list_rejection(self) -> None:
        """Assert caller with explicit allowed_tools cannot execute tools outside the allowlist."""
        registry = ToolRegistry()
        res = await registry.execute_async(
            "shell",
            allowed_tools={"echo"},
            command="whoami",
        )
        assert res.success is False
        assert "AuthorizationError" in str(res.error)
        assert "allowed tools list" in str(res.error)

    @pytest.mark.asyncio
    async def test_allowed_tools_list_acceptance(self) -> None:
        """Assert caller with explicit allowed_tools can execute tools within the allowlist."""
        registry = ToolRegistry()
        res = await registry.execute_async(
            "echo",
            allowed_tools={"echo"},
            text="valid allowlist execution",
        )
        assert res.success is True
        assert res.output == "valid allowlist execution"

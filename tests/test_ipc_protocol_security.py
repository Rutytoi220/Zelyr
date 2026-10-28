"""Tests for WebSocket IPC token authentication, JSON-RPC schema contracts, and rate limiting."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import stat
import pytest
import websockets

from zelyr.server.daemon import ZelyrDaemonServer
from zelyr.server.protocol import (
    INVALID_REQUEST,
    PARSE_ERROR,
    RATE_LIMIT_EXCEEDED,
    SlidingWindowRateLimiter,
)


def _get_free_port() -> int:
    """Find an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestIPCProtocolSecurity:
    """Verifies token authentication, schema contracts, rate limiting, and size boundaries."""

    def test_token_file_permissions(self, tmp_path: Path) -> None:
        """Assert runtime token file is created with strict 0600 POSIX permissions."""
        server = ZelyrDaemonServer()
        assert server.token_file.exists()
        mode = server.token_file.stat().st_mode
        file_perms = stat.S_IMODE(mode)
        assert file_perms == 0o600
        # Cleanup
        if server.token_file.exists():
            server.token_file.unlink()

    @pytest.mark.asyncio
    async def test_authentication_enforcement(self) -> None:
        """Assert connections without token or with invalid tokens are rejected with code 4401."""
        port = _get_free_port()
        server = ZelyrDaemonServer()
        await server.start(host="127.0.0.1", port=port, start_background=False)

        try:
            # 1. Connect without token
            with pytest.raises(websockets.exceptions.ConnectionClosed) as exc_info:
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.recv()
            assert exc_info.value.rcvd.code == 4401

            # 2. Connect with invalid token
            headers = {"Authorization": "Bearer completely_invalid_token"}
            with pytest.raises(websockets.exceptions.ConnectionClosed) as exc_info:
                async with websockets.connect(f"ws://127.0.0.1:{port}", additional_headers=headers) as ws:
                    await ws.recv()
            assert exc_info.value.rcvd.code == 4401

            # 3. Connect with valid Bearer token header
            valid_headers = {"Authorization": f"Bearer {server.auth_token}"}
            async with websockets.connect(f"ws://127.0.0.1:{port}", additional_headers=valid_headers) as ws:
                await ws.send(json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 1}))
                resp = json.loads(await ws.recv())
                assert resp["result"] == "pong"
                assert resp["id"] == 1

            # 4. Connect with valid query parameter ?token=<token>
            async with websockets.connect(f"ws://127.0.0.1:{port}/?token={server.auth_token}") as ws:
                await ws.send(json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 2}))
                resp = json.loads(await ws.recv())
                assert resp["result"] == "pong"
                assert resp["id"] == 2

        finally:
            await server.stop()
            assert not server.token_file.exists()

    @pytest.mark.asyncio
    async def test_contract_and_schema_validation(self) -> None:
        """Assert strict JSON-RPC 2.0 schema validation and standard error response envelopes."""
        port = _get_free_port()
        server = ZelyrDaemonServer()
        await server.start(host="127.0.0.1", port=port, start_background=False)

        try:
            valid_headers = {"Authorization": f"Bearer {server.auth_token}"}
            async with websockets.connect(f"ws://127.0.0.1:{port}", additional_headers=valid_headers) as ws:
                # 1. Valid JSON-RPC call
                await ws.send(json.dumps({"jsonrpc": "2.0", "method": "ping", "id": "req-99"}))
                resp = json.loads(await ws.recv())
                assert resp["jsonrpc"] == "2.0"
                assert resp["id"] == "req-99"
                assert resp["result"] == "pong"

                # 2. Malformed JSON parse error
                await ws.send("{broken_json: true,")
                err_resp = json.loads(await ws.recv())
                assert err_resp["jsonrpc"] == "2.0"
                assert err_resp["error"]["code"] == PARSE_ERROR
                assert "Parse error" in err_resp["error"]["message"]

                # 3. Missing required method/action fields
                await ws.send(json.dumps({"unexpected_field": "val"}))
                err_resp2 = json.loads(await ws.recv())
                assert err_resp2["jsonrpc"] == "2.0"
                assert err_resp2["error"]["code"] == INVALID_REQUEST

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_payload_size_ceiling_enforcement(self) -> None:
        """Assert payloads exceeding 1MB ceiling close connection with size violation code 4409 or 1009."""
        port = _get_free_port()
        server = ZelyrDaemonServer()
        # Allow websockets to receive the frame so daemon handler can inspect and emit 4409
        await server.start(host="127.0.0.1", port=port, start_background=False)

        try:
            valid_headers = {"Authorization": f"Bearer {server.auth_token}"}
            # Connect with high max_size so client can transmit the test payload
            async with websockets.connect(
                f"ws://127.0.0.1:{port}",
                additional_headers=valid_headers,
                max_size=2 * 1024 * 1024,
            ) as ws:
                oversized_payload = "A" * (1024 * 1024 + 1024)
                await ws.send(oversized_payload)
                with pytest.raises(websockets.exceptions.ConnectionClosed) as exc_info:
                    await ws.recv()
                assert exc_info.value.rcvd.code in (4409, 1009)

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_sliding_window_rate_limiting(self) -> None:
        """Assert burst flood exceeding rate limit returns rate limit error code -32029."""
        port = _get_free_port()
        server = ZelyrDaemonServer()
        await server.start(host="127.0.0.1", port=port, start_background=False)

        try:
            valid_headers = {"Authorization": f"Bearer {server.auth_token}"}
            async with websockets.connect(f"ws://127.0.0.1:{port}", additional_headers=valid_headers) as ws:
                rate_limited_count = 0
                success_count = 0

                # Send 65 rapid requests
                for i in range(65):
                    await ws.send(json.dumps({"jsonrpc": "2.0", "method": "ping", "id": i}))
                    resp_raw = await ws.recv()
                    resp = json.loads(resp_raw)
                    if "error" in resp and resp["error"].get("code") == RATE_LIMIT_EXCEEDED:
                        rate_limited_count += 1
                    elif "result" in resp:
                        success_count += 1

                # Confirms rate limiting kicked in after threshold
                assert rate_limited_count > 0
                assert success_count == 50

        finally:
            await server.stop()

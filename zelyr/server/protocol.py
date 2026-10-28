"""Protocol definitions, JSON-RPC 2.0 schemas, and rate limiting for Zelyr IPC."""

from __future__ import annotations

from collections import deque
import json
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator


MAX_PAYLOAD_SIZE = 1024 * 1024  # 1 MB message ceiling

# Standard JSON-RPC 2.0 Error Codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
RATE_LIMIT_EXCEEDED = -32029


class PayloadTooLargeError(Exception):
    """Raised when an incoming WebSocket frame or message payload exceeds 1 MB."""


class JSONRPCError(BaseModel):
    """Standard JSON-RPC 2.0 error object."""
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCRequest(BaseModel):
    """Strict JSON-RPC 2.0 request contract."""
    jsonrpc: str = Field(default="2.0")
    method: str = Field(min_length=1)
    params: Optional[Union[Dict[str, Any], List[Any]]] = None
    id: Optional[Union[str, int]] = None

    @field_validator("jsonrpc")
    @classmethod
    def validate_jsonrpc_version(cls, v: str) -> str:
        if v != "2.0":
            raise ValueError("jsonrpc version must be exactly '2.0'")
        return v


class JSONRPCResponse(BaseModel):
    """Standard JSON-RPC 2.0 response contract."""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None
    id: Optional[Union[str, int]] = None


def make_error_response(
    req_id: Optional[Union[str, int]],
    code: int,
    message: str,
    data: Optional[Any] = None,
) -> Dict[str, Any]:
    """Construct a standard JSON-RPC 2.0 error response dictionary."""
    err_obj: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err_obj["data"] = data
    return {
        "jsonrpc": "2.0",
        "error": err_obj,
        "id": req_id,
    }


def make_success_response(req_id: Optional[Union[str, int]], result: Any) -> Dict[str, Any]:
    """Construct a standard JSON-RPC 2.0 success response dictionary."""
    return {
        "jsonrpc": "2.0",
        "result": result,
        "id": req_id,
    }


class SlidingWindowRateLimiter:
    """Per-connection sliding window rate limiter enforcing a request ceiling."""

    def __init__(self, max_requests: int = 50, window_seconds: float = 1.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps: Deque[float] = deque()

    def allow_request(self, now: Optional[float] = None) -> bool:
        """Check if request is permitted under rate limit boundaries."""
        ts = time.time() if now is None else now
        while self.timestamps and ts - self.timestamps[0] > self.window_seconds:
            self.timestamps.popleft()

        if len(self.timestamps) >= self.max_requests:
            return False

        self.timestamps.append(ts)
        return True


def parse_and_validate_message(
    raw_message: str | bytes,
) -> Tuple[Optional[Union[JSONRPCRequest, Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """Inspect payload size, parse JSON, and validate against JSON-RPC / action contracts.

    Returns:
        (validated_request, None) on success
        (None, error_envelope_dict) on validation or parse failure
    """
    payload_len = len(raw_message.encode("utf-8") if isinstance(raw_message, str) else raw_message)
    if payload_len > MAX_PAYLOAD_SIZE:
        raise PayloadTooLargeError(
            f"Message size {payload_len} bytes exceeds maximum allowed ceiling ({MAX_PAYLOAD_SIZE} bytes)"
        )

    try:
        data = json.loads(raw_message)
    except Exception as exc:
        return None, make_error_response(None, PARSE_ERROR, f"Parse error: {exc}")

    if not isinstance(data, dict):
        return None, make_error_response(None, INVALID_REQUEST, "Invalid Request: root payload must be a JSON object")

    # If payload uses JSON-RPC 2.0 schema
    if "jsonrpc" in data or ("method" in data and "action" not in data):
        try:
            req = JSONRPCRequest.model_validate(data)
            return req, None
        except Exception as exc:
            req_id = data.get("id") if isinstance(data, dict) else None
            return None, make_error_response(req_id, INVALID_REQUEST, f"Invalid Request: {exc}")

    # Legacy action payload (e.g. {"action": "..."})
    if "action" in data and isinstance(data["action"], str) and data["action"].strip():
        return data, None

    return None, make_error_response(
        data.get("id"),
        INVALID_REQUEST,
        "Invalid Request: missing required 'method' (JSON-RPC) or 'action' field",
    )

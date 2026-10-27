"""Typed Data Models for the Zelyr SDK.

Provides strict Pydantic v2 schemas for JSON-RPC payloads and core
Zelyr domain entities (prompts, telemetry, swarm events).
"""

from typing import Any, Dict, List, Optional, Union  # pragma: no cover
from pydantic import BaseModel, Field  # pragma: no cover


class JsonRpcRequest(BaseModel):  # pragma: no cover
    """A standard JSON-RPC 2.0 Request."""
    jsonrpc: str = "2.0"  # pragma: no cover
    method: str
    params: Optional[Dict[str, Any]] = None  # pragma: no cover
    id: Optional[Union[str, int]] = None  # pragma: no cover


class JsonRpcError(BaseModel):  # pragma: no cover
    """A JSON-RPC 2.0 Error object."""
    code: int
    message: str
    data: Optional[Any] = None  # pragma: no cover


class JsonRpcResponse(BaseModel):  # pragma: no cover
    """A standard JSON-RPC 2.0 Response."""
    jsonrpc: str = "2.0"  # pragma: no cover
    result: Optional[Any] = None  # pragma: no cover
    error: Optional[JsonRpcError] = None  # pragma: no cover
    id: Optional[Union[str, int]] = None  # pragma: no cover


class PromptRequest(BaseModel):  # pragma: no cover
    """Payload for submitting a prompt to Zelyr."""
    text: str
    session_id: Optional[str] = None  # pragma: no cover


class TelemetryPayload(BaseModel):  # pragma: no cover
    """Telemetry payload emitted by Zelyr daemon."""
    ram_percent: float
    vram_percent: float
    tier: str


class SwarmProposal(BaseModel):  # pragma: no cover
    """A proposal emitted by a sub-agent within the swarm."""
    proposal_id: str
    tool: str
    agent: str
    status: str = "PROPOSED"  # pragma: no cover


class SwarmVote(BaseModel):  # pragma: no cover
    """A vote cast by a sub-agent regarding a proposal."""
    proposal_id: str
    vote: str
    voter: str

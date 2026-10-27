"""Zelyr SDK - Official Python Client Library.

Provides asynchronous and synchronous clients, as well as strict
Pydantic v2 data models for interacting with the Zelyr JSON-RPC daemon.
"""

from zelyr.sdk.client import ZelyrClient, SyncZelyrClient  # pragma: no cover

# Legacy names retained for existing AXIOM integrations.
AxiomClient = ZelyrClient
SyncAxiomClient = SyncZelyrClient
from zelyr.sdk.models import (  # pragma: no cover
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    PromptRequest,
    TelemetryPayload,
    SwarmProposal,
    SwarmVote,
)

__all__ = [  # pragma: no cover
    "ZelyrClient",
    "SyncZelyrClient",
    "AxiomClient",
    "SyncAxiomClient",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "JsonRpcError",
    "PromptRequest",
    "TelemetryPayload",
    "SwarmProposal",
    "SwarmVote",
]

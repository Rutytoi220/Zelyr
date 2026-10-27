# Zelyr SDK Developer Guide

The `zelyr-sdk` is the official Python client library for integrating with the Zelyr JSON-RPC 2.0 daemon. It provides both asynchronous and synchronous wrappers around the core Unix Domain Socket API, featuring automatic authentication, strict Pydantic typings, and event streaming.

## Installation

Since Zelyr SDK is bundled with the OS environment, you can import it directly from your Python scripts running on the system:

```python
from zelyr.sdk import ZelyrClient, SyncZelyrClient
from zelyr.sdk import PromptRequest, TelemetryPayload
```

## Quick Start (Async)

The primary interface is asynchronous, which is ideal for high-throughput or event-driven applications like custom IDE extensions.

```python
import asyncio
from zelyr.sdk import ZelyrClient

async def main():
    # Automatically locates ~/.zelyr/zelyr.sock and ~/.zelyr/daemon.token
    client = ZelyrClient()
    await client.connect()

    # 1. Check System Status
    status = await client.get_status()
    print(f"System Load: RAM {status.ram_percent}%, VRAM {status.vram_percent}%")

    # 2. Submit a Prompt
    result = await client.prompt("Analyze the current directory and summarize its purpose.")
    print("Zelyr response:", result)
    
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

## Quick Start (Sync Wrapper)

For simple scripts, use `SyncZelyrClient`. This wrapper runs an internal background thread and event loop, allowing you to seamlessly integrate with synchronous code.

```python
from zelyr.sdk import SyncZelyrClient

client = SyncZelyrClient()

# Blocks until the response is ready
status = client.get_status()
print(status.model_dump())

result = client.prompt("Hello Zelyr!")
print(result)
```

## EventBus Streaming

You can subscribe directly to the daemon's EventBus to observe the inner workings of Zelyr, including Multi-Agent Swarm proposals, system state changes, and proactive perception events.

```python
import asyncio
from zelyr.sdk import ZelyrClient

async def monitor_swarm():
    client = ZelyrClient()
    await client.connect()
    
    print("Listening for sub-agent proposals...")
    
    # Subscribe to a specific topic
    async for event in client.subscribe("swarm.proposal"):
        print(f"New Proposal [{event['proposal_id']}]: Agent {event['agent']} wants to run {event['tool']}")

if __name__ == "__main__":
    asyncio.run(monitor_swarm())
```

### Common Event Topics

- `swarm.proposal`: When a sub-agent proposes an action.
- `swarm.vote`: When consensus votes are cast.
- `telemetry.update`: Hardware RAM/VRAM tracking.
- `perception.window.change`: Emitted by the Proactive OS Kernel (if enabled).
- `perception.clipboard.change`: Emitted when new clipboard text passes the privacy scrubber.

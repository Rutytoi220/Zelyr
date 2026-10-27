# ADR-002: Async-Aware EventBus with Wildcard Pattern Matching

**Status:** Accepted  
**Date:** 2026-07-18  
**Authors:** Zelyr Core Team

---

## Context

Zelyr is a modular AI orchestration framework composed of loosely coupled subsystems — agents, tools, plugins, memory compaction daemons, and telemetry pipelines — that must communicate without direct dependencies on each other. The framework operates as both an interactive desktop application (via PySide6/qasync) and a headless background daemon, often simultaneously.

We evaluated three approaches for inter-component communication:

1. **Direct synchronous function calls.** Simple, but creates tight coupling. Adding a new telemetry listener would require modifying the engine core. Unacceptable for a plugin-extensible architecture.
2. **Heavy message brokers (RabbitMQ, Redis Pub/Sub, ZeroMQ).** Battle-tested at scale, but introduces an external service dependency. This directly violates Zelyr's local-first, zero-dependency philosophy. A user should not need to run a Redis server to use a desktop AI assistant.
3. **A pure-Python in-process pub/sub EventBus.** Zero external dependencies, tight integration with the Python runtime, and full control over dispatch semantics.

The critical constraint was that Zelyr's GUI event loop (`qasync` wrapping `asyncio`) and its background daemon both needed to publish and consume events without blocking either runtime.

## Decision

We built a custom, single-class `EventBus` ([events.py](file:///home/rutytoi/Documents/ChienGPT/zelyr/core/events.py)) that is both synchronous-safe and async-aware:

### Core Design

```python
class EventBus:
    def subscribe(self, event_type: str, handler: Callable) -> None: ...
    def publish(self, event: Event) -> None: ...
    def publish_sync(self, event_name: str, data: Any = None) -> None: ...
```

### Key Engineering Decisions

**1. Hybrid Sync/Async Dispatch**  
The `publish()` method probes for a running `asyncio` event loop at call time. If one exists, coroutine handlers are dispatched via `loop.create_task()` and blocking handlers via `loop.run_in_executor(None, ...)`. If no loop is running (pure CLI mode), handlers are called synchronously. This allows the same EventBus instance to serve both the GUI daemon and the synchronous CLI REPL without configuration changes.

**2. `fnmatch` Wildcard Pattern Subscriptions**  
Subscribers can register with glob-style patterns (`agent.*`, `*.error`, `*`). The matching engine uses Python's `fnmatch` module. This enables the `SleepCycleDaemon` to subscribe to `*` for idle-detection, the flight recorder to capture `tool.*` events, and plugins to listen to domain-specific namespaces — all without the bus needing to know about them at compile time.

**3. TTL-Based Event Storm Prevention**  
Each `Event` carries a `ttl: int` field (default 5). The bus drops events with `ttl <= 0`, preventing infinite re-emission loops when handlers publish derivative events. This is a lightweight alternative to circuit breakers.

**4. Hash-Based Debouncing**  
High-frequency telemetry (thermal sensors, power governor polls) can flood the bus. A debounce cache keyed on `f"{event_type}:{source}:{hash(data)}"` with a configurable `_debounce_ttl` (default 2.0s) silently drops duplicate events, reducing CPU overhead to near zero during idle polling.

**5. Meta-Event Observability**  
After every publish cycle, the bus emits a `bus.published` meta-event (re-entrancy guarded via `_in_meta_event` flag). This allows observability tooling to monitor bus throughput without intercepting every event type individually.

### Data Model

```python
@dataclass
class Event:
    event_type: str          # Dotted namespace (e.g. "agent.task.completed")
    source: str              # Originating component
    timestamp: datetime      # Auto-populated
    event_id: str            # UUID v4
    correlation_id: str      # UUID v4 for request tracing
    ttl: int = 5             # Time-to-live for storm prevention
    data: Dict[str, Any]     # Payload
    metadata: Dict[str, Any] # Extensible metadata bag
```

## Consequences

### Positive

- **Zero external dependencies.** No Redis, no RabbitMQ, no Docker containers. The bus is a single 185-line Python file.
- **GIL-friendly async dispatch.** By offloading blocking handlers to `run_in_executor`, CPU-bound work executes in OS threads outside the GIL while the asyncio loop remains unblocked. The GUI never freezes due to event processing.
- **Wildcard subscriptions eliminate tight coupling.** The `SleepCycleDaemon`, `OllamaHealthMonitor`, `ProfileService`, and all plugins can subscribe independently. Adding a new subsystem requires zero changes to the bus.
- **Event storm resilience.** The TTL + debounce combination has proven robust in production under heavy telemetry loads.
- **Full history introspection.** The last 1,000 events are retained in-memory, enabling the `/trace --last` CLI command and the Flight Recorder telemetry replay system.

### Negative

- **Async stack traces are harder to debug.** When a handler is dispatched via `create_task()`, exceptions surface as unhandled task exceptions in the asyncio event loop rather than at the `publish()` call site. This requires disciplined `try/except` wrapping inside every handler.
- **No delivery guarantees.** Unlike a proper message broker, there is no persistent queue, no retry policy, and no dead-letter mechanism. If a handler crashes, the event is lost. This is acceptable for Zelyr's use case (desktop telemetry), but would not be for a distributed microservice.
- **In-process only.** The bus cannot span multiple OS processes. The daemon-to-GUI bridge uses a separate Unix Domain Socket (UDS) JSON-RPC protocol for IPC, which must manually re-publish events across the boundary.

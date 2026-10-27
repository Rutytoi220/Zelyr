# Zelyr Architecture Decision Records (ADRs)

Architecture Decision Records capture the key engineering tradeoffs made during Zelyr's development. They follow the [Michael Nygard ADR format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions): each record documents the **Context** (the problem), the **Decision** (what we chose and why), and the **Consequences** (the good and bad tradeoffs).

ADRs are immutable once accepted. If a decision is later reversed, a new ADR is created referencing the superseded record.

---

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-002](ADR-002-async-eventbus.md) | Async-Aware EventBus with Wildcard Pattern Matching | Accepted | 2026-07-18 |
| [ADR-003](ADR-003-sqlite-wal-memory.md) | SQLite WAL + Local Vector Database for Semantic Memory | Accepted | 2026-07-18 |
| [ADR-004](ADR-004-dynamic-plugin-sdk.md) | Dynamic Plugin SDK with Manifest-Based Discovery and Sandboxed Execution | Accepted | 2026-07-18 |

---

## How to Add a New ADR

1. Create a new file: `ADR-NNN-short-title.md`
2. Follow the standard format: **Title**, **Status**, **Date**, **Context**, **Decision**, **Consequences**
3. Add an entry to this index table
4. Submit a PR for team review

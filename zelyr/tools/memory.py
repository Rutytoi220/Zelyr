"""Zelyr Memory Tools — Autonomous Fact Retention."""

from typing import Any, Dict
from zelyr.tools.core import BaseTool, ToolResult
from zelyr.memory.semantic import add_memory


class RememberFactTool(BaseTool):
    """Tool allowing the model to autonomously store important user preferences,
    system configurations, and persistent project details into long-term memory.
    """

    def __init__(self):
        super().__init__()
        self._name = "remember_fact"
        self._description = (
            "Store an important fact, user preference, system configuration, or persistent project detail "
            "into long-term memory."
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The concise fact, user preference, or project rule to store (alias for content)."
                },
                "content": {
                    "type": "string",
                    "description": "The memory content to store."
                },
                "category": {
                    "type": "string",
                    "enum": ["preference", "environment", "homelab", "workflow", "capability"],
                    "description": "The category of the memory (default: preference)."
                },
                "key": {
                    "type": "string",
                    "description": "Optional unique key for contradiction resolution and superseding (e.g. user.shell)."
                }
            },
            "required": []
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        fact = (params.get("content") or params.get("fact") or "").strip()
        if not fact:
            return ToolResult(success=False, error="Parameter 'fact' or 'content' cannot be empty.")

        category = params.get("category", "preference")
        if category not in ("preference", "environment", "homelab", "workflow", "capability"):
            category = "preference"

        key = params.get("key") or None

        try:
            from zelyr.db.memory import MemoryStore
            store = MemoryStore()
            mem_id = store.store_fact(content=fact, category=category, key=key)

            # Optionally attach semantic embedding if generator is available
            try:
                from zelyr.memory.semantic import generate_embedding, serialize_embedding
                emb = generate_embedding(fact)
                if emb:
                    blob = serialize_embedding(emb)
                    with store._get_conn() as conn:
                        conn.execute("UPDATE memories SET embedding = ? WHERE id = ?", (blob, mem_id))
                        conn.commit()
            except Exception:
                pass

            return ToolResult(
                success=True,
                output={
                    "message": f"Successfully stored memory #{mem_id}: {fact}",
                    "memory_id": mem_id,
                    "fact": fact,
                    "content": fact,
                    "category": category,
                    "key": key,
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to store memory: {e}")


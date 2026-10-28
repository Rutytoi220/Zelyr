from zelyr.core.events import Event
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

class DeepMemoryConsolidation:
    """
    Background maintenance daemon for Zelyr GraphRAG Memory.
    Mimics biological REM sleep to deduplicate, fuse, and decay memory graphs
    without locking the SQLite WAL while PySide6 UI is active.
    """

    def __init__(self, event_bus):
        self.event_bus = event_bus

    async def trigger_rem_sleep(self, graph_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate, fuse, and decay memory graph nodes asynchronously.
        Heavy yielding ensures the SQLite database isn't locked.
        
        Args:
            graph_nodes: A list of dicts representing memory nodes.
                         Each node typically has 'id', 'text' or 'content', 'timestamp' or 'created_at'.
                         
        Returns:
            A consolidated list of memory nodes after decay and deduplication.
        """
        if not graph_nodes:
            return []

        logger.info(f"[REM Sleep] Starting deep memory consolidation on {len(graph_nodes)} nodes.")
        self.event_bus.publish(
            Event(event_type="rem_sleep.started", source="REM", data={})
        )

        consolidated = []
        seen_hashes = {}
        decay_threshold = datetime.now(timezone.utc) - timedelta(days=30)
        
        pruned_count = 0
        fused_count = 0

        for idx, node in enumerate(graph_nodes):
            # Yield every 10 nodes to avoid locking SQLite or blocking the event loop
            if idx % 10 == 0:
                await asyncio.sleep(0.01)

            # 1. Temporal Decay Scoring
            node_time_str = node.get("created_at") or node.get("timestamp")
            
            # Parse timestamp if available, default to now if missing
            node_time = datetime.now(timezone.utc)
            if isinstance(node_time_str, str):
                try:
                    # Very basic ISO parsing
                    from dateutil.parser import parse
                    node_time = parse(node_time_str)
                    if node_time.tzinfo is None:
                        node_time = node_time.replace(tzinfo=timezone.utc)
                except Exception:
                    pass
            elif isinstance(node_time_str, (int, float)):
                node_time = datetime.fromtimestamp(node_time_str, timezone.utc)

            # Check protected categories & core beliefs
            category = (node.get("category") or "").lower()
            key = (node.get("key") or "").lower()
            tags = [str(t).lower() for t in node.get("tags", [])] if isinstance(node.get("tags"), list) else []
            is_protected = (
                node.get("is_core", False)
                or node.get("core_belief", False)
                or "core_belief" in tags
                or "core_belief" in key
                or category in {"preference", "environment", "capability", "core", "workflow", "homelab"}
            )

            if not is_protected and node_time < decay_threshold:
                # Ephemeral log decay: drop it
                pruned_count += 1
                continue

            # 2. Semantic Deduplication
            content = node.get("text") or node.get("content") or node.get("value", "")
            if isinstance(content, str):
                # Normalize and fuse
                normalized_content = " ".join(content.strip().lower().split())
                
                # We can create a simple semantic hash
                content_hash = hash(normalized_content)
                
                if content_hash in seen_hashes:
                    # Fuse node: keep the oldest/most connected node, inherit retrieval count and boost confidence
                    original_idx = seen_hashes[content_hash]
                    fused_count += 1
                    
                    surviving = consolidated[original_idx]
                    cur_rc = surviving.get("retrieval_count") or surviving.get("access_count") or 0
                    dup_rc = node.get("retrieval_count") or node.get("access_count") or 0
                    surviving["retrieval_count"] = max(cur_rc, dup_rc) + 1
                    surviving["access_count"] = surviving["retrieval_count"]

                    cur_conf = float(surviving.get("confidence") or surviving.get("confidence_weight") or 1.0)
                    dup_conf = float(node.get("confidence") or node.get("confidence_weight") or 1.0)
                    surviving["confidence"] = min(1.0, max(cur_conf, dup_conf) + 0.1)
                    surviving["confidence_weight"] = surviving["confidence"]
                    continue
                else:
                    seen_hashes[content_hash] = len(consolidated)
            
            consolidated.append(node)

        logger.info(
            f"[REM Sleep] Completed. Pruned {pruned_count} decayed nodes. "
            f"Fused {fused_count} duplicates. Surviving nodes: {len(consolidated)}."
        )
        
        self.event_bus.publish(
            Event(
                event_type="rem_sleep.completed",
                source="REM",
                data={"surviving": len(consolidated), "pruned": pruned_count, "fused": fused_count},
            )
        )

        return consolidated


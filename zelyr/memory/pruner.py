"""Memory Pruner Daemon for Zelyr.

Runs a daily sweep to summarize and compress old messages, and run VACUUM on the database.
"""
import time
import logging
import asyncio
import json
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

class MemoryPrunerDaemon:
    """Runs an hourly check to prune, summarize, and vacuum the memory database based on quota limits."""

    def __init__(self, memory_store: Any, llm: Optional[Any] = None):
        self._memory_store = memory_store
        self._llm = llm
        self._task: Optional[asyncio.Task] = None
        self._is_running = False
        self._stop_event: Optional[asyncio.Event] = None
        self._active_db: Optional[Any] = None
        self._db_mgr: Optional[Any] = None

    def start(self) -> None:
        if self._is_running:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("No running asyncio event loop; MemoryPrunerDaemon will not start in synchronous context.")
            return

        self._is_running = True
        self._stop_event = asyncio.Event()
        self._task = loop.create_task(self._monitor_loop())
        logger.info("MemoryPrunerDaemon started (interval: 3600s)")

    def stop(self) -> None:
        """Signal the daemon loop to stop."""
        self._is_running = False
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()

    async def shutdown(self, timeout: float = 3.0) -> None:
        """Authoritative asynchronous teardown of the daemon and its resources.

        a. Signals the daemon loop to stop.
        b. Cancels running _monitor_loop task.
        c. Awaits task cancellation using asyncio.wait_for:
           - On timeout: retains self._task, raises TimeoutError, keeps resources unclosed.
           - Handles CancelledError normally when task concludes cancellation.
        d. Only after task completion is verified (self._task.done()):
           - Explicitly closes any open SQLite database connections, cursors, or file descriptors.
           - Flushes WAL checkpoints where applicable.
           - Clears task references.
        """
        self._is_running = False
        if self._stop_event is not None:
            self._stop_event.set()

        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
            except (asyncio.TimeoutError, TimeoutError):
                raise TimeoutError(f"Daemon shutdown timed out after {timeout}s; monitor task still running.")
            except asyncio.CancelledError:
                if not self._task.done():
                    raise TimeoutError(f"Daemon shutdown timed out after {timeout}s; monitor task still running.")
            except Exception as exc:
                logger.debug(f"Daemon task exited with exception: {exc}")

        if self._task is not None and not self._task.done():
            return

        # Verified completed task -> clean up held resources
        if self._active_db is not None:
            try:
                await self._active_db.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception:
                pass
            try:
                await self._active_db.close()
            except Exception:
                pass
            self._active_db = None

        if self._db_mgr is not None:
            self._db_mgr = None

        self._task = None
        logger.info("MemoryPrunerDaemon shutdown complete.")

    def close(self) -> None:
        """Strict synchronous teardown.

        Raises RuntimeError if an active event loop is currently running.
        Otherwise executes shutdown() via asyncio.run().
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError(
                "Cannot call synchronous close() while an event loop is running. Await daemon.shutdown() instead."
            )

        asyncio.run(self.shutdown())

    async def _monitor_loop(self) -> None:
        try:
            if self._stop_event is not None:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=5)
                    return
                except (asyncio.TimeoutError, TimeoutError):
                    pass
            else:
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            return

        while self._is_running:
            logger.info("MemoryPrunerDaemon: Running prune sweep...")
            try:
                await self._prune_and_vacuum()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"MemoryPrunerDaemon failed during prune cycle: {e}", exc_info=True)

            try:
                if self._stop_event is not None:
                    try:
                        await asyncio.wait_for(self._stop_event.wait(), timeout=3600)
                        break
                    except (asyncio.TimeoutError, TimeoutError):
                        pass
                else:
                    await asyncio.sleep(3600)
            except asyncio.CancelledError:
                break

    async def _prune_and_vacuum(self) -> None:
        from zelyr.config import get_config
        config = get_config()
        quota = getattr(config, 'max_vector_memories', 5000)

        # Get the database path dynamically from the store
        db_path = self._memory_store.store.db_path if hasattr(self._memory_store, "store") else self._memory_store.db_path
        from zelyr.memory.db import MemoryDatabaseManager
        db_mgr = await MemoryDatabaseManager.get_instance(db_path)
        db = await db_mgr.get_connection()
        self._db_mgr = db_mgr
        self._active_db = db
        
        # 1. Count current memories
        async with db.execute("SELECT COUNT(id) FROM messages") as cursor:
            row = await cursor.fetchone()
            total_memories = row[0] if row else 0

        if total_memories <= quota:
            logger.info(f"MemoryPrunerDaemon: Usage ({total_memories}) is under quota ({quota}).")
            return

        excess = total_memories - quota
        to_prune = max(1000, excess)
        logger.warning(f"MemoryPrunerDaemon: Quota exceeded! ({total_memories} > {quota}). Pruning {to_prune} oldest memories.")

        import aiosqlite
        db.row_factory = aiosqlite.Row

        # 2. Identify oldest episodic memories
        async with db.execute(
            "SELECT id, conversation_id, role, content, timestamp FROM messages ORDER BY timestamp ASC LIMIT ?",
            (to_prune,)
        ) as cursor:
            rows = await cursor.fetchall()
        
        if not rows:
            return

        conversations = {}
        for row in rows:
            c_id = row["conversation_id"]
            if c_id not in conversations:
                conversations[c_id] = []
            conversations[c_id].append(row)
        
        for c_id, msgs in conversations.items():
            transcript = []
            msg_ids_to_delete = []
            for m in msgs:
                role = m["role"].upper() if m["role"] else "UNKNOWN"
                content = m["content"] or ""
                transcript.append(f"{role}: {content}")
                msg_ids_to_delete.append(m["id"])
            
            summary = "[Archived Context: Dense Summary generated by Pruner]"
            if self._llm:
                prompt = (
                    "Summarize the following old chat messages into a single dense context block. "
                    "Focus on key facts, decisions, and important information. Keep it concise.\n\n" +
                    "\n".join(transcript)
                )
                messages = [{"role": "user", "content": prompt}]
                response = None
                try:
                    if hasattr(self._llm, "chat_with_tools"):
                        response_msg = await asyncio.to_thread(self._llm.chat_with_tools, messages, [], 60.0)
                        response = response_msg.get("content", "") if isinstance(response_msg, dict) else str(response_msg)
                    elif hasattr(self._llm, "chat"):
                        response = await asyncio.to_thread(self._llm.chat, messages, 60.0)
                    
                    if response:
                        summary = response.strip()
                except Exception as e:
                    logger.error(f"MemoryPrunerDaemon: LLM summarization failed: {e}")
            
            # 3. Purge and Replace
            sys_msg_id = uuid.uuid4().hex
            cutoff_time = msgs[-1]["timestamp"]
            formatted_date = time.strftime("%Y-%m-%d", time.localtime(cutoff_time))
            sys_content = f"[Archived Context before {formatted_date}]\n{summary}"
            
            # Safe Context Managers for Execution
            async with db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, metadata_json, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (sys_msg_id, c_id, "system", sys_content, json.dumps({"pruned_messages": len(msgs)}), cutoff_time)
            ) as _:
                pass
            
            for chunk in [msg_ids_to_delete[i:i + 100] for i in range(0, len(msg_ids_to_delete), 100)]:
                placeholders = ",".join("?" * len(chunk))
                async with db.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", chunk) as _:
                    pass
            
        await db.commit()
        logger.info("MemoryPrunerDaemon: Purge complete.")
        
        logger.info("MemoryPrunerDaemon: Running VACUUM on database...")
        try:
            async with db.execute("VACUUM") as _:
                pass
            logger.info("MemoryPrunerDaemon: VACUUM completed successfully.")
        except Exception as e:
            logger.error(f"MemoryPrunerDaemon: VACUUM failed: {e}")


"""Unit and integration test suite for Phase 3:
Session Promotion, Contradiction Handling & Capability Indexing.

Verifies:
1. Tool Capability Auto-Indexing:
   - Auto-indexing loads tool schemas and indexes into category 'capability' with key 'tool.<name>'.
   - Modifying schema description supersedes old active record atomically.
   - Idempotent re-runs skip identical capabilities without duplicate rows.
2. Contradiction & Superseding Resolution:
   - MemoryStore.store_fact() automatically infers key for domain patterns (shell, editor, theme, etc.).
   - Explicit or inferred key updates supersede prior active records (is_active=0, superseded_by=new_id).
3. Modernized Memory Tools (remember_fact & search_memory):
   - RememberFactTool supports both 'content' and legacy 'fact' arguments, category, and key.
   - SearchMemoryTool retrieves relevant memories from MemoryStore with category filtering and historical fallback.
4. Session Memory Promotion Engine:
   - Scans user statements ("remember that", "always use", "my preference is") and assistant remember_fact calls.
   - Stores new facts in long-term memory under proper categories.
   - Idempotent execution on same session messages.
5. REPL do_exit Integration:
   - Graceful teardown in InlineRepl triggers session promotion.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path

from zelyr.db.memory import (
    MemoryStore,
    init_db,
    create_session,
    add_message,
    get_session_messages,
)
from zelyr.memory.capability_indexer import index_tool_capabilities
from zelyr.memory.promotion import promote_session_facts
from zelyr.tools.memory import RememberFactTool
from zelyr.cli.repl import InlineRepl


class TestMemoryPhase3(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "test_memory_p3.db")
        self.store = MemoryStore(db_path=self.db_path)
        os.environ["ZELYR_MEMORY_DB"] = self.db_path

    def tearDown(self):
        self.temp_dir.cleanup()
        if "ZELYR_MEMORY_DB" in os.environ:
            del os.environ["ZELYR_MEMORY_DB"]

    def test_1_capability_indexing_and_superseding(self):
        """Verify tool capabilities are indexed under 'capability' and superseded on change."""
        dummy_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "mock_docker_tool",
                    "description": "Inspect and manage docker containers v1.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mock_journal_tool",
                    "description": "Query systemd journal logs.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        # 1. Initial indexing
        indexed = index_tool_capabilities(store=self.store, schemas=dummy_schemas)
        self.assertEqual(indexed, 2)

        # Verify records created
        rec1 = self.store.get_by_key("tool.mock_docker_tool")
        self.assertIsNotNone(rec1)
        self.assertEqual(rec1["category"], "capability")
        self.assertEqual(rec1["content"], "mock_docker_tool: Inspect and manage docker containers v1.")
        self.assertEqual(rec1["is_active"], 1)

        rec2 = self.store.get_by_key("tool.mock_journal_tool")
        self.assertIsNotNone(rec2)
        self.assertEqual(rec2["content"], "mock_journal_tool: Query systemd journal logs.")

        # 2. Idempotent re-run should skip unchanged schemas
        indexed_repeat = index_tool_capabilities(store=self.store, schemas=dummy_schemas)
        self.assertEqual(indexed_repeat, 0)

        # 3. Update description of mock_docker_tool -> should supersede
        updated_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "mock_docker_tool",
                    "description": "Inspect and manage docker containers v2 with compose support.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            dummy_schemas[1],
        ]

        indexed_updated = index_tool_capabilities(store=self.store, schemas=updated_schemas)
        self.assertEqual(indexed_updated, 1)

        # Verify active record is now v2
        active_rec1 = self.store.get_by_key("tool.mock_docker_tool")
        self.assertIsNotNone(active_rec1)
        self.assertEqual(
            active_rec1["content"],
            "mock_docker_tool: Inspect and manage docker containers v2 with compose support.",
        )
        self.assertEqual(active_rec1["is_active"], 1)

        # Verify old record was superseded
        old_rec1 = self.store.get(rec1["id"])
        self.assertEqual(old_rec1["is_active"], 0)
        self.assertEqual(old_rec1["superseded_by"], active_rec1["id"])

    def test_2_contradiction_handling_and_key_inference(self):
        """Verify contradiction superseding with explicit keys and inferred keys."""
        # 1. Explicit key superseding
        id1 = self.store.store_fact(
            content="User prefers zsh with starship",
            category="preference",
            key="user.shell",
        )
        self.assertGreater(id1, 0)

        id2 = self.store.store_fact(
            content="User prefers fish shell",
            category="preference",
            key="user.shell",
        )
        self.assertGreater(id2, id1)

        # Check that id1 is superseded by id2
        rec_id1 = self.store.get(id1)
        self.assertEqual(rec_id1["is_active"], 0)
        self.assertEqual(rec_id1["superseded_by"], id2)

        active_shell = self.store.get_by_key("user.shell")
        self.assertEqual(active_shell["id"], id2)
        self.assertEqual(active_shell["content"], "User prefers fish shell")

        # 2. Key inference: editor domain
        id3 = self.store.store_fact(
            content="Preferred editor is Neovim with Lua plugins",
            category="preference",
        )
        rec_id3 = self.store.get(id3)
        self.assertEqual(rec_id3["key"], "user.editor")

        # Contradiction with inferred key
        id4 = self.store.store_fact(
            content="User switched to VSCode editor",
            category="preference",
        )
        rec_id4 = self.store.get(id4)
        self.assertEqual(rec_id4["key"], "user.editor")

        rec_id3_updated = self.store.get(id3)
        self.assertEqual(rec_id3_updated["is_active"], 0)
        self.assertEqual(rec_id3_updated["superseded_by"], id4)

    async def test_3_modernized_remember_fact_tool(self):
        """Verify RememberFactTool works with content/fact, category, and key parameters."""
        tool = RememberFactTool()

        # 1. Backwards compatible 'fact' parameter
        res1 = await tool.execute({"fact": "Always run pytest with -v flag"})
        self.assertTrue(res1.success)
        self.assertIn("message", res1.output)
        self.assertTrue(res1.output["message"].startswith("Successfully stored memory #"))
        self.assertIn("memory_id", res1.output)
        self.assertEqual(res1.output["fact"], "Always run pytest with -v flag")
        self.assertEqual(res1.output["content"], "Always run pytest with -v flag")

        # 2. Modern 'content', 'category', and 'key' parameters
        res2 = await tool.execute({
            "content": "Postgres runs on port 5432 with sslmode=require",
            "category": "homelab",
            "key": "homelab.db.postgres",
        })
        self.assertTrue(res2.success)
        self.assertEqual(res2.output["category"], "homelab")
        self.assertEqual(res2.output["key"], "homelab.db.postgres")

        # 3. Supersede via key
        res3 = await tool.execute({
            "content": "Postgres runs on port 5433 (new cluster)",
            "category": "homelab",
            "key": "homelab.db.postgres",
        })
        self.assertTrue(res3.success)

        id_old = res2.output["memory_id"]
        id_new = res3.output["memory_id"]
        rec_old = self.store.get(id_old)
        self.assertEqual(rec_old["is_active"], 0)
        self.assertEqual(rec_old["superseded_by"], id_new)

    async def test_4_modernized_search_memory_tool(self):
        """Verify search_memory tool wrapper queries MemoryStore and supports category filter."""
        from importlib.machinery import SourceFileLoader
        tools_path = Path.home() / ".config" / "zelyr" / "tools.d" / "search_memory.py"
        if not tools_path.exists():
            self.skipTest("search_memory.py not found in ~/.config/zelyr/tools.d")

        mod = SourceFileLoader("search_memory_tool", str(tools_path)).load_module()

        # Populate memory store
        self.store.store(
            content="Primary DNS is 1.1.1.1 and fallback is 8.8.8.8",
            category="environment",
            key="network.dns",
        )
        self.store.store(
            content="Git workflow requires conventional commits",
            category="workflow",
            key="workflow.git.commits",
        )

        # Search without category filter
        output_dns = await mod.execute(query="DNS fallback")
        self.assertIn("1.1.1.1", output_dns)
        self.assertIn("network.dns", output_dns)

        # Search with matching category
        output_wf = await mod.execute(query="commits", category="workflow")
        self.assertIn("workflow.git.commits", output_wf)

        # Search with non-matching category
        output_none = await mod.execute(query="commits", category="homelab")
        self.assertEqual(output_none, "No relevant memories found.")

    def test_5_session_fact_promotion(self):
        """Verify promote_session_facts extracts user preferences and directives."""
        session_id = "test_promo_session_123"
        create_session(session_id, title="Test Promotion Session", db_path=self.db_path)

        # Add message history with explicit user directives
        add_message(session_id, "user", "Hello Zelyr, please remember that my shell is zsh.", db_path=self.db_path)
        add_message(session_id, "assistant", "Got it, I will remember that you use zsh.", db_path=self.db_path)
        add_message(session_id, "user", "Also, always use git rebase instead of merge.", db_path=self.db_path)
        add_message(session_id, "assistant", "Noted, git rebase will always be used.", db_path=self.db_path)
        add_message(session_id, "user", "Note that postgres database runs on host 10.0.0.5.", db_path=self.db_path)

        # Run promotion
        promoted = promote_session_facts(session_id, store=self.store)
        self.assertGreater(len(promoted), 0)

        # Verify promoted facts exist in memory store
        all_active = [self.store.get(pid) for pid in promoted]
        contents = [m["content"] for m in all_active if m]

        # Check contents
        self.assertTrue(any("zsh" in c for c in contents), f"zsh not found in {contents}")
        self.assertTrue(any("git rebase" in c for c in contents), f"git rebase not found in {contents}")
        self.assertTrue(any("postgres" in c for c in contents), f"postgres not found in {contents}")

        # Check categories
        categories = {m["category"] for m in all_active if m}
        self.assertTrue("preference" in categories or "workflow" in categories or "homelab" in categories)

        # Idempotency check: running again on same messages creates no duplicate rows
        promoted_second = promote_session_facts(session_id, store=self.store)
        self.assertEqual(len(promoted_second), 0)

    async def test_6_repl_do_exit_triggers_promotion(self):
        """Verify InlineRepl.do_exit() promotes facts from the active session."""
        session_id = "test_repl_do_exit_session"
        create_session(session_id, title="REPL Teardown Test", db_path=self.db_path)
        add_message(session_id, "user", "Remember that my editor is neovim.", db_path=self.db_path)

        repl = InlineRepl(session_id=session_id)
        # Execute teardown
        await repl.do_exit()

        # Check if memory was promoted into memory store
        active_editor = self.store.get_by_key("user.editor")
        self.assertIsNotNone(active_editor)
        self.assertIn("neovim", active_editor["content"].lower())


if __name__ == "__main__":
    unittest.main()

"""Test suite for SQLite FTS5 persistent memory store and prompt hydration."""

import os
import tempfile
import threading
import unittest
from pathlib import Path

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.db.memory import (
    VALID_CATEGORIES,
    MemoryStore,
    add_message,
    create_session,
    get_connection,
    get_session_messages,
    search_historical_context,
)
from zelyr.memory.hydration import format_contextual_memory, hydrate_context_memory


class TestSqliteMemoryStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "test_memory_p1.db")
        self.store = MemoryStore(db_path=self.db_path)
        os.environ["ZELYR_MEMORY_DB"] = self.db_path

    def tearDown(self):
        self.temp_dir.cleanup()
        if "ZELYR_MEMORY_DB" in os.environ:
            del os.environ["ZELYR_MEMORY_DB"]

    def test_crud_lifecycle(self):
        """Verify full lifecycle: store, get, update, soft-delete, purge, hard-delete."""
        mem_id = self.store.store(
            content="The user prefers dark mode and high contrast themes.",
            category="preference",
            key="ui.theme",
            confidence=0.95,
        )
        self.assertIsInstance(mem_id, int)
        self.assertGreater(mem_id, 0)

        mem = self.store.get(mem_id)
        self.assertIsNotNone(mem)
        self.assertEqual(mem["id"], mem_id)
        self.assertEqual(mem["category"], "preference")
        self.assertEqual(mem["key"], "ui.theme")
        self.assertEqual(mem["content"], "The user prefers dark mode and high contrast themes.")
        self.assertEqual(mem["confidence"], 0.95)
        self.assertEqual(mem["is_active"], 1)
        self.assertIsNone(mem["superseded_by"])
        self.assertEqual(mem["access_count"], 1)

        # Second read increments access count
        mem_again = self.store.get(mem_id)
        self.assertEqual(mem_again["access_count"], 2)

        by_key = self.store.get_by_key("ui.theme")
        self.assertIsNotNone(by_key)
        self.assertEqual(by_key["id"], mem_id)

        updated = self.store.update(
            mem_id,
            content="The user prefers Tokyo Night dark mode theme.",
            confidence=1.0,
        )
        self.assertTrue(updated)
        mem_up = self.store.get(mem_id)
        self.assertEqual(mem_up["content"], "The user prefers Tokyo Night dark mode theme.")
        self.assertEqual(mem_up["confidence"], 1.0)
        self.assertGreaterEqual(mem_up["updated_at"], mem["updated_at"])

        deleted = self.store.delete(mem_id, hard_delete=False)
        self.assertTrue(deleted)

        mem_soft = self.store.get(mem_id)
        self.assertIsNotNone(mem_soft)
        self.assertEqual(mem_soft["is_active"], 0)
        self.assertIsNone(self.store.get_by_key("ui.theme"))

        # search excludes soft-deleted by default
        search_active = self.store.search("Tokyo Night")
        self.assertEqual(len(search_active), 0)

        # search includes when include_inactive=True
        search_inactive = self.store.search("Tokyo Night", include_inactive=True)
        self.assertEqual(len(search_inactive), 1)
        self.assertEqual(search_inactive[0]["id"], mem_id)

        purged = self.store.purge_inactive()
        self.assertEqual(purged, 1)
        self.assertIsNone(self.store.get(mem_id))

        new_id = self.store.store(content="Temporary memory to hard delete", category="capability")
        self.assertTrue(self.store.delete(new_id, hard_delete=True))
        self.assertIsNone(self.store.get(new_id))

    def test_key_automatic_superseding(self):
        """Storing a new active memory with the same key must atomically supersede old version."""
        id_v1 = self.store.store(
            content="User uses bash as default shell",
            category="environment",
            key="user.shell",
            confidence=0.8,
        )
        mem_v1 = self.store.get(id_v1)
        self.assertEqual(mem_v1["is_active"], 1)
        self.assertIsNone(mem_v1["superseded_by"])

        id_v2 = self.store.store(
            content="User switched to zsh with starship prompt",
            category="environment",
            key="user.shell",
            confidence=0.99,
        )
        self.assertNotEqual(id_v1, id_v2)

        mem_v1_after = self.store.get(id_v1)
        self.assertEqual(mem_v1_after["is_active"], 0)
        self.assertEqual(mem_v1_after["superseded_by"], id_v2)

        mem_v2 = self.store.get(id_v2)
        self.assertEqual(mem_v2["is_active"], 1)
        self.assertIsNone(mem_v2["superseded_by"])

        active_shell = self.store.get_by_key("user.shell")
        self.assertIsNotNone(active_shell)
        self.assertEqual(active_shell["id"], id_v2)
        self.assertEqual(active_shell["content"], "User switched to zsh with starship prompt")

    def test_fts5_search_synchronization(self):
        """Verify FTS5 search across content, key, and category with triggers."""
        self.store.store(
            content="Proxmox VE hypervisor running on 192.168.1.100",
            category="homelab",
            key="homelab.proxmox",
        )
        self.store.store(
            content="TrueNAS Core storage server on 192.168.1.105",
            category="homelab",
            key="homelab.nas",
        )
        self.store.store(
            content="Git workflow requires conventional commits and PR squash merge",
            category="workflow",
            key="workflow.git",
        )

        res_proxmox = self.store.search("Proxmox")
        self.assertEqual(len(res_proxmox), 1)
        self.assertEqual(res_proxmox[0]["key"], "homelab.proxmox")

        res_key = self.store.search("homelab.nas")
        self.assertEqual(len(res_key), 1)
        self.assertEqual(res_key[0]["key"], "homelab.nas")

        res_cat = self.store.search("homelab")
        self.assertGreaterEqual(len(res_cat), 2)
        keys = {r["key"] for r in res_cat}
        self.assertIn("homelab.proxmox", keys)
        self.assertIn("homelab.nas", keys)

        res_workflow = self.store.search("commits", category="workflow")
        self.assertEqual(len(res_workflow), 1)
        self.assertEqual(res_workflow[0]["category"], "workflow")

        res_empty = self.store.search("commits", category="homelab")
        self.assertEqual(len(res_empty), 0)

    def test_category_validation(self):
        """Store must reject invalid categories not defined in CHECK constraint."""
        for cat in VALID_CATEGORIES:
            mem_id = self.store.store(content=f"Valid test for {cat}", category=cat)
            self.assertGreater(mem_id, 0)

        with self.assertRaises(ValueError):
            self.store.store(content="Invalid category test", category="unsupported_cat")

    def test_wal_concurrency(self):
        """Simulate concurrent threads reading and writing simultaneously without DB lock errors."""
        errors = []
        threads = []
        num_threads = 4
        writes_per_thread = 15

        def worker(thread_idx: int):
            try:
                for i in range(writes_per_thread):
                    key = f"thread_{thread_idx}.item_{i % 5}"
                    mem_id = self.store.store(
                        content=f"Thread {thread_idx} written item {i} with payload data",
                        category="capability",
                        key=key,
                        confidence=1.0,
                    )

                    mem = self.store.get(mem_id)
                    if not mem or mem["id"] != mem_id:
                        errors.append(f"Thread {thread_idx}: get({mem_id}) failed")

                    by_key = self.store.get_by_key(key)
                    if not by_key or by_key["key"] != key:
                        errors.append(f"Thread {thread_idx}: get_by_key({key}) failed")

                    search_res = self.store.search(f"item {i}")
                    if not search_res:
                        errors.append(f"Thread {thread_idx}: search('item {i}') returned empty")

            except Exception as e:
                errors.append(f"Thread {thread_idx} exception: {e}")

        for t_idx in range(num_threads):
            t = threading.Thread(target=worker, args=(t_idx,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrency errors encountered: {errors}")

        conn = get_connection(self.db_path)
        total_active = conn.execute("SELECT count(*) FROM memories WHERE is_active = 1").fetchone()[0]
        self.assertGreater(total_active, 0)
        conn.close()

    def test_chat_session_history_backwards_compatibility(self):
        """Verify sessions, messages, and messages_fts work without regression."""
        session_id = "test_backwards_session_1"
        create_session(session_id, title="Test Session", db_path=self.db_path)

        add_message(session_id, "user", "What is the status of the server?", db_path=self.db_path)
        add_message(session_id, "assistant", "All systems operational with 99.9% uptime.", db_path=self.db_path)

        msgs = get_session_messages(session_id, db_path=self.db_path)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "What is the status of the server?")
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertEqual(msgs[1]["content"], "All systems operational with 99.9% uptime.")

        fts_search = search_historical_context("operational", session_id, db_path=self.db_path)
        self.assertEqual(len(fts_search), 1)
        self.assertIn("All systems operational", fts_search[0])


class TestMemoryHydration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "test_hydration_p2.db")
        self.store = MemoryStore(db_path=self.db_path)
        os.environ["ZELYR_MEMORY_DB"] = self.db_path

    async def asyncTearDown(self):
        self.temp_dir.cleanup()
        if "ZELYR_MEMORY_DB" in os.environ:
            del os.environ["ZELYR_MEMORY_DB"]

    def test_query_matching_and_category_formatting(self):
        """Verify matched memories are formatted cleanly into [Contextual Memory] block."""
        self.store.store(
            content="zsh with starship prompt",
            category="preference",
            key="user.shell",
            confidence=1.0,
        )
        self.store.store(
            content="192.168.1.100 proxmox node",
            category="homelab",
            key="server.ip",
            confidence=0.9,
        )
        self.store.store(
            content="Always use git rebase on main branch",
            category="workflow",
            key=None,
            confidence=0.95,
        )

        block_shell = hydrate_context_memory("What user.shell is configured?", store=self.store)
        self.assertIn("[Contextual Memory]:", block_shell)
        self.assertIn("<recalled_memory>", block_shell)
        self.assertIn("</recalled_memory>", block_shell)
        self.assertIn("- (preference) user.shell: zsh with starship prompt", block_shell)
        self.assertNotIn("proxmox", block_shell)

        block_server = hydrate_context_memory("server.ip proxmox node", store=self.store)
        self.assertIn("- (homelab) server.ip: 192.168.1.100 proxmox node", block_server)

        block_workflow = hydrate_context_memory("rebase main branch", store=self.store)
        self.assertIn("- (workflow) Always use git rebase on main branch", block_workflow)

    def test_hard_token_ceiling_with_high_volume_matching(self):
        """Enforce strict ceiling of <= 350 tokens under heavy memory volume."""
        for i in range(30):
            self.store.store(
                content=f"Kubernetes cluster node {i} configuration parameters with distributed raft consensus logging and telemetry enabled on homelab subnet {i}",
                category="homelab",
                key=f"cluster.node_{i}",
                confidence=1.0,
            )

        relevant = self.store.get_relevant_memories("cluster homelab", max_tokens=350)
        self.assertGreater(len(relevant), 0)
        self.assertLess(len(relevant), 15)

        formatted = format_contextual_memory(relevant)
        token_count = NativeOrchestrator.count_tokens(formatted)
        self.assertLessEqual(token_count, 350, f"Token count {token_count} exceeds 350 ceiling!")
        self.assertGreater(token_count, 50)
        self.assertIn("[Contextual Memory]:", formatted)
        self.assertIn("<recalled_memory>", formatted)

    def test_zero_token_injection_for_empty_or_irrelevant_query(self):
        """When query is empty or completely irrelevant, zero tokens are injected."""
        self.store.store(
            content="Postgres database port is 5432",
            category="environment",
            key="db.port",
        )

        self.assertEqual(hydrate_context_memory("", store=self.store), "")
        self.assertEqual(hydrate_context_memory("    ", store=self.store), "")
        self.assertEqual(hydrate_context_memory("quantum_superposition_xyz_9999", store=self.store), "")

    def test_access_count_increments_on_injection(self):
        """Only memories actually selected and injected have their access_count incremented."""
        id1 = self.store.store(content="Primary editor is Neovim", category="preference", key="user.editor")
        id2 = self.store.store(content="Secondary machine is Raspberry Pi", category="homelab", key="lab.rpi")

        # Initially 1 from get()
        self.assertEqual(self.store.get(id1)["access_count"], 1)
        self.assertEqual(self.store.get(id2)["access_count"], 1)

        # Query matching only id1
        _ = self.store.get_relevant_memories("Neovim editor", max_tokens=350, update_access=True)

        mem1 = self.store.get(id1)
        mem2 = self.store.get(id2)
        # id1 was injected and retrieved: access_count increased
        self.assertGreaterEqual(mem1["access_count"], 3)
        # id2 was NOT injected, its access_count only incremented from our get() call
        self.assertEqual(mem2["access_count"], 2)

    async def test_native_orchestrator_depth0_hydration_integration(self):
        """Verify NativeOrchestrator integrates hydration on depth 0 and bypasses on depth > 0, retry, or recovery."""
        self.store.store(
            content="zsh with starship prompt",
            category="preference",
            key="user.shell",
            confidence=1.0,
        )

        orch = NativeOrchestrator()

        # Case 1: Standard depth == 0 turn -> Should hydrate
        payload_standard = {
            "messages": [{"role": "user", "content": "What is my favorite user.shell?"}],
            "tools": [],
        }
        async for _ in orch.generate_stream(payload_standard, depth=0, is_retry=False):
            break

        sys_msg = payload_standard["messages"][0]["content"]
        self.assertIn("[Contextual Memory]:", sys_msg)
        self.assertIn("- (preference) user.shell: zsh with starship prompt", sys_msg)

        # Case 2: depth == 1 -> Should NOT hydrate
        payload_depth1 = {
            "messages": [
                {"role": "system", "content": "Base instructions."},
                {"role": "user", "content": "What is my favorite user.shell?"},
            ],
            "tools": [],
        }
        async for _ in orch.generate_stream(payload_depth1, depth=1, is_retry=False):
            break
        sys_msg_d1 = payload_depth1["messages"][0]["content"]
        self.assertNotIn("[Contextual Memory]:", sys_msg_d1)

        # Case 3: is_retry == True -> Should NOT hydrate
        payload_retry = {
            "messages": [{"role": "user", "content": "What is my favorite user.shell?"}],
            "tools": [],
        }
        async for _ in orch.generate_stream(payload_retry, depth=0, is_retry=True):
            break
        sys_msg_retry = payload_retry["messages"][0]["content"]
        self.assertNotIn("[Contextual Memory]:", sys_msg_retry)

        # Case 4: Preceding turn is tool error recovery turn -> Should NOT hydrate
        payload_recovery = {
            "messages": [
                {"role": "user", "content": "Click the submit button"},
                {"role": "assistant", "content": "Calling click_element", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "interact_with_browser"}}]},
                {"role": "tool", "content": "[Tool Error]: Element not found. Recovery Hint: switch tab"}
            ],
            "tools": [],
        }
        async for _ in orch.generate_stream(payload_recovery, depth=0, is_retry=False):
            break
        sys_msg_recovery = payload_recovery["messages"][0]["content"]
        self.assertNotIn("[Contextual Memory]:", sys_msg_recovery)


if __name__ == "__main__":
    unittest.main()

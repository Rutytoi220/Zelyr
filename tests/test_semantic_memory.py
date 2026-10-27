"""Comprehensive verification test harness for:
1. Hierarchical Custom Instructions (global instructions.md + workspace ZELYR.md/.zelyrrules)
2. Context Window Budgeting & Sliding Window Pruning
3. Persistent Semantic Vector Memory & Memory Injection
4. REPL Memory & Rules Slash Commands
"""

import asyncio
import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from zelyr.core.instructions import InstructionManager, SYSTEM_CORE_DIRECTIVES
from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.memory.semantic import (
    add_memory,
    search_memories,
    get_all_memories,
    generate_embedding,
    serialize_embedding,
    deserialize_embedding,
    cosine_similarity,
    bm25_search,
    SemanticMemoryStore,
)
from zelyr.tools.memory import RememberFactTool
from zelyr.core.plugins import execute_tool, get_tool_schemas
from zelyr.cli.repl import InlineRepl, COMMANDS


def test_1_hierarchical_instructions():
    print("=" * 60)
    print("[CHECK 1] Testing Hierarchical Instruction Engine...")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        global_file = tmp_path / "global_instructions.md"
        global_file.write_text("# Global Rules\nUser prefers strict type safety.", encoding="utf-8")

        project_dir = tmp_path / "my_project"
        project_dir.mkdir()
        zelyr_md = project_dir / "ZELYR.md"
        zelyr_md.write_text("# Workspace Rules\nNever modify .env directly.", encoding="utf-8")

        # 1. Test InstructionManager loading both layers
        im = InstructionManager(global_path=global_file, cwd=project_dir)
        global_rules = im.get_global_instructions()
        assert "User prefers strict type safety." in global_rules, f"Unexpected global rules: {global_rules}"

        proj_rules = im.get_project_instructions()
        assert "Never modify .env directly." in proj_rules, f"Unexpected project rules: {proj_rules}"

        # 2. Test compiled instructions format
        compiled = im.get_compiled_instructions()
        assert "[SYSTEM CORE DIRECTIVES]" in compiled
        assert "[USER CUSTOM INSTRUCTIONS]" in compiled
        assert "User prefers strict type safety." in compiled
        assert "[PROJECT WORKSPACE RULES]" in compiled
        assert "Never modify .env directly." in compiled
        print("  → Compiled instructions format validated.")

        # 3. Test .zelyrrules alternative project file
        zelyr_md.unlink()
        zelyr_rules = project_dir / ".zelyrrules"
        zelyr_rules.write_text("# Alt Rules\nAlways use git rebase.", encoding="utf-8")
        assert "Always use git rebase." in im.get_project_instructions()
        print("  → Alternative .zelyrrules detection validated.")

        # 4. Test missing global template creation
        missing_global = tmp_path / "sub" / "missing_instructions.md"
        im_missing = InstructionManager(global_path=missing_global, cwd=project_dir)
        template_content = im_missing.get_global_instructions()
        assert missing_global.is_file(), "Failed to create missing template file."
        assert "Zelyr Global Custom Instructions" in template_content
        print("  → Auto-generation of default template validated.")

        # 5. Test NativeOrchestrator._build_system_prompt()
        orch = NativeOrchestrator()
        sys_prompt = orch._build_system_prompt()
        assert "[SYSTEM CORE DIRECTIVES]" in sys_prompt
        assert "[USER CUSTOM INSTRUCTIONS]" in sys_prompt
        assert "[PROJECT WORKSPACE RULES]" in sys_prompt
        print("  → NativeOrchestrator._build_system_prompt() integration validated.")

    print("✓ Check 1 PASSED: Hierarchical instructions work cleanly.\n")


def test_2_context_sliding_window():
    print("=" * 60)
    print("[CHECK 2] Testing Context Window Budgeting & Sliding Window...")
    print("=" * 60)

    # Reserve 1000 tokens; max_context_tokens = 2000 -> conversation budget = 1000 tokens
    orch = NativeOrchestrator(max_context_tokens=2000)

    system_msg = {"role": "system", "content": "System directives here."}
    turn_1_user = {"role": "user", "content": "Initial user task: set up the backend."}
    turn_1_assist = {"role": "assistant", "content": "Backend setup started."}

    # Bulky turns in the middle (~400 tokens each)
    turn_2_user = {"role": "user", "content": "Bulky log data turn 2: " + ("data " * 300)}
    turn_2_assist = {"role": "assistant", "content": "Bulky response turn 2: " + ("ok " * 300)}

    turn_3_user = {"role": "user", "content": "Bulky log data turn 3: " + ("data " * 300)}
    turn_3_assist = {"role": "assistant", "content": "Bulky response turn 3: " + ("ok " * 300)}

    # Latest turns (~50 tokens each)
    turn_4_user = {"role": "user", "content": "Latest user request: check test results."}

    messages = [
        system_msg,
        turn_1_user,
        turn_1_assist,
        turn_2_user,
        turn_2_assist,
        turn_3_user,
        turn_3_assist,
        turn_4_user,
    ]

    total_tokens_before = sum(orch.count_message_tokens(m) for m in messages if m["role"] != "system")
    print(f"  → Total conversation tokens before pruning: {total_tokens_before} (budget: 1000)")
    assert total_tokens_before > 1000, "Test setup error: initial tokens should exceed budget."

    pruned = orch.prune_messages(messages, max_context_tokens=2000, reserve_tokens=1000)
    conv_pruned = [m for m in pruned if m["role"] != "system"]
    total_tokens_after = sum(orch.count_message_tokens(m) for m in conv_pruned)
    print(f"  → Total conversation tokens after pruning: {total_tokens_after}")

    assert total_tokens_after <= 1000, f"Pruning failed to respect budget: {total_tokens_after} > 1000"

    # Invariants verification:
    # 1. System directive preserved
    assert pruned[0]["role"] == "system"
    assert pruned[0]["content"] == "System directives here."

    # 2. Initial non-system turn preserved
    assert conv_pruned[0] == turn_1_user, "Initial turn was lost during pruning."

    # 3. Latest turn preserved
    assert conv_pruned[-1] == turn_4_user, "Latest turn was lost during pruning."

    # 4. Oldest middle turn was pruned
    assert turn_2_user not in conv_pruned, "Oldest turn 2 was not pruned."
    assert turn_2_assist not in conv_pruned, "Oldest turn 2 response was not pruned."

    print("✓ Check 2 PASSED: Context sliding window prunes middle turns and respects budget.\n")


def test_3_semantic_memory_engine():
    print("=" * 60)
    print("[CHECK 3] Testing Persistent Semantic Memory & Vector Search...")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
        db_path = tmp_db.name

        store = SemanticMemoryStore(db_path=db_path)

        # 1. Add memories
        id1 = store.add_memory("The user prefers Tokyo Night theme and Neovim editor.")
        id2 = store.add_memory("The project database credentials are in /etc/zelyr/secrets.env.")
        id3 = store.add_memory("The user's dog is named Pixel and loves chasing laser pointers.")

        assert id1 > 0 and id2 > 0 and id3 > 0, "Failed to insert memories."
        all_mems = store.get_all_memories()
        assert len(all_mems) == 3, f"Expected 3 memories, got {len(all_mems)}"
        print(f"  → Successfully stored 3 persistent memories in SQLite (IDs: {id1}, {id2}, {id3}).")

        # 2. Search relevant memory
        results_editor = store.search_memories("What text editor does the user like?", top_k=1)
        print(f"  → Query 'What text editor does the user like?': {results_editor}")
        assert len(results_editor) > 0, "No search results returned."
        assert "Neovim" in results_editor[0], f"Expected Neovim in results, got: {results_editor}"

        results_pet = store.search_memories("Tell me about the pet dog", top_k=1)
        print(f"  → Query 'Tell me about the pet dog': {results_pet}")
        assert len(results_pet) > 0
        assert "Pixel" in results_pet[0]

        # 3. Verify BM25 fallback search directly
        docs = [(1, "App runs on port 8080"), (2, "Postgres port is 5432"), (3, "Redis cache port is 6379")]
        bm25_res = bm25_search("Where does postgres run?", docs, top_k=1)
        assert bm25_res == ["Postgres port is 5432"], f"BM25 search mismatch: {bm25_res}"
        print("  → BM25 lexical fallback verified.")

    print("✓ Check 3 PASSED: Semantic memory storage and similarity retrieval verified.\n")


@pytest.mark.asyncio
async def test_4_tool_and_memory_injection():
    print("=" * 60)
    print("[CHECK 4] Testing remember_fact Tool & Prompt Memory Injection...")
    print("=" * 60)

    # 1. Test remember_fact tool execution
    fact_text = "The user prefers zsh shell with starship prompt."
    res_str = await execute_tool("remember_fact", fact=fact_text)
    res_data = json.loads(res_str)
    assert res_data.get("success") is True, f"remember_fact failed: {res_str}"
    print(f"  → remember_fact tool executed successfully: {res_data['result']['output']}")

    # 2. Test memory recall injection into prompt payload
    orch = NativeOrchestrator()
    payload = {
        "messages": [
            {"role": "user", "content": "What shell and prompt does the user prefer?"}
        ],
        "tools": [],
    }

    # Consume first stream chunk to trigger system prompt building and memory injection
    async for chunk in orch.generate_stream(payload):
        break

    # Inspect messages in payload
    sys_content = payload["messages"][0]["content"]
    assert "<recalled_memory>" in sys_content, f"Expected <recalled_memory> in system prompt, got: {sys_content[:400]}"
    assert "zsh" in sys_content or "starship" in sys_content, "Recalled fact content missing from injected block."
    print("  → Memory recall successfully injected <recalled_memory> into prompt payload.")

    print("✓ Check 4 PASSED: remember_fact tool and autonomous memory recall verified.\n")


def test_5_repl_slash_commands():
    print("=" * 60)
    print("[CHECK 5] Testing REPL /memory and /rules Slash Commands...")
    print("=" * 60)

    assert "/memory" in COMMANDS, "/memory missing from COMMANDS"
    assert "/rules" in COMMANDS, "/rules missing from COMMANDS"

    repl = InlineRepl(session_id="test_memory_repl_session")

    # 1. Test /rules
    handled_rules = repl.handle_command("/rules")
    assert handled_rules is True, "/rules command failed to execute."

    # 2. Test /memory add
    handled_add = repl.handle_command("/memory add Testing persistent REPL memory entry.")
    assert handled_add is True, "/memory add command failed to execute."

    # 3. Test /memory list
    handled_list = repl.handle_command("/memory list")
    assert handled_list is True, "/memory list command failed to execute."

    print("✓ Check 5 PASSED: REPL slash commands for memory and rules verified.\n")


async def main():
    test_1_hierarchical_instructions()
    test_2_context_sliding_window()
    test_3_semantic_memory_engine()
    await test_4_tool_and_memory_injection()
    test_5_repl_slash_commands()

    print("=" * 60)
    print("✅ ALL HIERARCHICAL INSTRUCTION, CONTEXT BUDGETING & SEMANTIC MEMORY CHECKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

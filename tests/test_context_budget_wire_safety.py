"""Comprehensive test suite for Context Window Budgeting and Active-Turn Wire Preservation.

Verifies:
1. Atomic ReAct turn block identification:
   - Tool calls and matching tool observations form indivisible units.
   - Blocks are never severed into orphaned tool results or unanswered tool calls.
2. Inviolate anchors under extreme budget pressure:
   - System prompt (messages[0]) is strictly immune.
   - Root user goal (first user message) is strictly immune.
   - Active user turn + all in-progress ReAct steps are strictly immune to eviction.
3. Historical observation compaction:
   - Non-active tool messages > 1,500 chars are compacted with head 500 and tail 500 chars.
   - Structured JSON envelope metadata (status, exit_code, error_type, remedy_hint) preserved.
   - Active turn tool messages are preserved untouched.
4. Wire-format validation & post-pruning sanitization:
   - Every tool response is preceded by an assistant tool-caller.
   - Dangling/orphaned tool messages are eliminated.
   - Unanswered tool calls are safely synthesized or pruned so wire protocol never fails.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from zelyr.agents.context_budgeter import (
    BudgetEngine,
    ContextBudgetEngine,
    count_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    group_into_atomic_turns,
    sanitize_wire_invariants,
)
from zelyr.agents.native_orchestrator import NativeOrchestrator


class TestAtomicReActTurnGrouping:
    """Tests for _group_into_atomic_turns indivisible ReAct turn grouping."""

    def test_group_into_atomic_turns_basic_and_multitool(self):
        """Verify tool calls and corresponding tool responses stay grouped together."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are Zelyr."},
            {"role": "user", "content": "Check disk and memory."},
            {
                "role": "assistant",
                "content": "Executing diagnostics.",
                "tool_calls": [
                    {"id": "call_disk_1", "type": "function", "function": {"name": "df"}},
                    {"id": "call_mem_1", "type": "function", "function": {"name": "free"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_disk_1", "name": "df", "content": "Filesystem /dev/sda1 80% used"},
            {"role": "tool", "tool_call_id": "call_mem_1", "name": "free", "content": "Mem: 16Gi total, 8Gi free"},
            {"role": "assistant", "content": "Diagnostics complete. Disk is 80% full, Memory is 50% free."},
        ]

        blocks = group_into_atomic_turns(messages)

        # Expected 4 atomic blocks:
        # Block 0: [system]
        # Block 1: [user] (Wait: user is followed by assistant with tool calls, so user is [user])
        # Block 2: [assistant, tool_disk, tool_mem] (ReAct step block)
        # Block 3: [assistant final]
        assert len(blocks) == 4
        assert len(blocks[0]) == 1 and blocks[0][0]["role"] == "system"
        assert len(blocks[1]) == 1 and blocks[1][0]["role"] == "user"

        react_block = blocks[2]
        assert len(react_block) == 3
        assert react_block[0]["role"] == "assistant"
        assert react_block[1]["role"] == "tool" and react_block[1]["tool_call_id"] == "call_disk_1"
        assert react_block[2]["role"] == "tool" and react_block[2]["tool_call_id"] == "call_mem_1"

        final_block = blocks[3]
        assert len(final_block) == 1 and final_block[0]["role"] == "assistant"

        # Flattened blocks must exactly match original messages sequence
        flattened = [m for b in blocks for m in b]
        assert flattened == messages

    def test_group_user_and_simple_assistant_response(self):
        """User message followed by simple assistant message (no tools) pairs into turn block."""
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language."},
        ]
        blocks = group_into_atomic_turns(messages)
        assert len(blocks) == 1
        assert len(blocks[0]) == 2
        assert blocks[0][0]["role"] == "user"
        assert blocks[0][1]["role"] == "assistant"


class TestHistoricalToolCompaction:
    """Tests for Stage 1 historical observation compaction (>1500 chars)."""

    def test_historical_observation_compaction_head_tail_500(self):
        """Historical tool outputs > 1500 chars are compacted with head 500 + tail 500 and metadata preserved."""
        engine = ContextBudgetEngine(tool_output_token_limit=250, recent_turns_to_preserve=1)

        # Bulky observation > 1500 characters
        head_sample = "START_TRACE_LOG_LINE_" * 25  # ~525 chars
        middle_noise = "EXCESS_DEBUG_BUFFER_DATA_CHUNK_" * 100  # ~3100 chars
        tail_sample = "END_METRIC_REPORT_SUMMARY_" * 20  # ~520 chars
        large_content = head_sample + middle_noise + tail_sample
        assert len(large_content) > 4000

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are Zelyr."},
            # Turn 1 (Historical)
            {"role": "user", "content": "Run deep diagnostics on network."},
            {"role": "assistant", "content": "Scanning network...", "tool_calls": [{"id": "c1", "name": "scan"}]},
            {"role": "tool", "tool_call_id": "c1", "name": "scan", "content": large_content},
            {"role": "assistant", "content": "Scan complete."},
            # Turn 2 (Active Turn: latest user query)
            {"role": "user", "content": "Run quick ping."},
            {"role": "assistant", "content": "Pinging host...", "tool_calls": [{"id": "c2", "name": "ping"}]},
            {"role": "tool", "tool_call_id": "c2", "name": "ping", "content": large_content},
        ]

        compacted, count = engine.stage1_compact_tool_results(
            messages,
            recent_turns_to_preserve=1,
            tool_output_token_limit=250,
        )

        assert count == 1  # Only Turn 1 tool was compacted

        # Turn 1 tool observation must be compacted
        t1_tool = compacted[3]
        assert t1_tool["role"] == "tool"
        assert "[... output truncated:" in t1_tool["content"]
        assert "TRUNCATED FOR CONTEXT BUDGET" in t1_tool["content"]
        assert head_sample[:500] in t1_tool["content"]
        assert tail_sample[-500:] in t1_tool["content"]

        # Turn 2 tool observation (Active Turn) must remain completely UNTOUCHED
        t2_tool = compacted[7]
        assert t2_tool["role"] == "tool"
        assert t2_tool["content"] == large_content
        assert "[... output truncated:" not in t2_tool["content"]
        assert "TRUNCATED FOR CONTEXT BUDGET" not in t2_tool["content"]

    def test_json_metadata_preservation_under_compaction(self):
        """Compaction must preserve JSON metadata keys (tool, status, exit_code, error_type, remedy_hint)."""
        engine = ContextBudgetEngine(tool_output_token_limit=250)

        bulky_stdout = "DUMP_ROW_" * 300  # ~2700 chars
        raw_json_tool = json.dumps({
            "tool": "verify_outcome",
            "status": "failure",
            "exit_code": 1,
            "error_type": "AssertionError",
            "remedy_hint": "Check file path permissions",
            "stdout": bulky_stdout,
        })

        compacted_text = engine.compact_tool_output(raw_json_tool, token_limit=250, keep_head_chars=500, keep_tail_chars=500)
        parsed = json.loads(compacted_text)

        assert parsed["tool"] == "verify_outcome"
        assert parsed["status"] == "failure"
        assert parsed["exit_code"] == 1
        assert parsed["error_type"] == "AssertionError"
        assert parsed["remedy_hint"] == "Check file path permissions"
        assert "[... output truncated:" in parsed["stdout"]
        assert "TRUNCATED FOR CONTEXT BUDGET" in parsed["stdout"]


class TestActiveTurnAndAnchorImmunity:
    """Tests guaranteeing system prompt, root user goal, and active turn immunity."""

    def test_active_turn_and_root_goal_immunity_under_extreme_budget(self):
        """Under severe token budget pressure (e.g. 500 tokens), root goal and active turn survive intact."""
        engine = ContextBudgetEngine()

        sys_msg = {"role": "system", "content": "You are Zelyr autonomous assistant."}
        root_user = {"role": "user", "content": "ROOT TASK: Refactor authentication module."}
        turn1_assist = {"role": "assistant", "content": "Refactoring auth...", "tool_calls": [{"id": "c1", "name": "edit"}]}
        turn1_tool = {"role": "tool", "tool_call_id": "c1", "name": "edit", "content": "edit ok"}
        turn1_done = {"role": "assistant", "content": "Auth refactored."}

        # Middle bulky turn
        turn2_user = {"role": "user", "content": "Run tests on auth: " + ("test_spec " * 150)}
        turn2_assist = {"role": "assistant", "content": "Tests executed: " + ("pass_spec " * 150)}

        # Active turn in progress with ReAct step
        active_user = {"role": "user", "content": "ACTIVE QUERY: Verify database migration."}
        active_assist = {
            "role": "assistant",
            "content": "Checking migration status.",
            "tool_calls": [{"id": "c_mig_1", "type": "function", "function": {"name": "db_check"}}],
        }
        active_tool = {
            "role": "tool",
            "tool_call_id": "c_mig_1",
            "name": "db_check",
            "content": "Migration 0042 applied cleanly.",
        }

        messages = [
            sys_msg,
            root_user,
            turn1_assist,
            turn1_tool,
            turn1_done,
            turn2_user,
            turn2_assist,
            active_user,
            active_assist,
            active_tool,
        ]

        # Enforce severe budget of 600 tokens
        pruned, evicted_count = engine.stage2_prune_turns(messages, target_budget=600)

        assert evicted_count > 0

        # Invariant 1: System prompt is at index 0
        assert pruned[0] == sys_msg

        # Invariant 2: Summary anchor exists immediately after system prompt
        assert pruned[1]["role"] == "system"
        assert "[Previous conversation summary:" in pruned[1]["content"]

        # Invariant 3: Root User Goal is strictly preserved
        assert root_user in pruned

        # Invariant 4: Middle bulky turn 2 is evicted
        assert turn2_user not in pruned
        assert turn2_assist not in pruned

        # Invariant 5: Active Turn (user query + in-progress ReAct assistant & tool) strictly preserved
        assert active_user in pruned
        assert active_assist in pruned
        assert active_tool in pruned
        assert pruned[-2] == active_assist
        assert pruned[-1] == active_tool


class TestWireSafetySanitization:
    """Tests for _sanitize_wire_invariants and orphan prevention."""

    def test_orphan_tool_discarded_by_sanitizer(self):
        """Orphaned tool message (not preceded by assistant) is safely discarded."""
        corrupted_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User prompt"},
            {"role": "tool", "tool_call_id": "orphan_call_1", "name": "orphan_tool", "content": "stray result"},
            {"role": "assistant", "content": "Final response"},
        ]

        sanitized = sanitize_wire_invariants(corrupted_messages)

        # Stray tool must be dropped
        roles = [m["role"] for m in sanitized]
        assert "tool" not in roles
        assert roles == ["system", "user", "assistant"]

    def test_unanswered_tool_call_synthesized(self):
        """Assistant message with declared tool_call_id lacking response is synthesized to prevent 400 wire error."""
        unanswered_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Execute tool"},
            {
                "role": "assistant",
                "content": "Executing...",
                "tool_calls": [
                    {"id": "call_unanswered_1", "type": "function", "function": {"name": "query_db"}},
                ],
            },
            # Missing tool message; immediately followed by next user message
            {"role": "user", "content": "Next turn query"},
        ]

        sanitized = sanitize_wire_invariants(unanswered_messages)

        # A synthetic tool message must have been injected right after assistant
        assert len(sanitized) == 5
        assert sanitized[2]["role"] == "assistant"
        assert sanitized[3]["role"] == "tool"
        assert sanitized[3]["tool_call_id"] == "call_unanswered_1"
        assert sanitized[4]["role"] == "user"

    def test_atomic_pruning_never_creates_orphans(self):
        """Under aggressive budget pruning, assistant tool-calls and tool messages never get severed."""
        orchestrator = NativeOrchestrator(max_context_tokens=1500)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are Zelyr."},
            {"role": "user", "content": "Root goal: system check."},
        ]

        # Add 6 turns of tool executions with moderate payloads
        for i in range(1, 7):
            tc_id = f"call_{i}"
            messages.append({"role": "user", "content": f"Turn {i} request."})
            messages.append({
                "role": "assistant",
                "content": f"Running step {i}",
                "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": "exec_cmd"}}],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": "exec_cmd",
                "content": f"Output step {i}: " + ("data " * 100),
            })
            messages.append({"role": "assistant", "content": f"Step {i} completed."})

        # Prune down to tight budget
        pruned = orchestrator.prune_messages(messages, max_context_tokens=1500, reserve_tokens=300)

        # Verify wire invariant: every tool message is preceded by assistant with tool_calls
        for idx, msg in enumerate(pruned):
            if msg.get("role") == "tool":
                assert idx > 0, "Tool message cannot be at index 0"
                prev = pruned[idx - 1]
                assert prev.get("role") in ("assistant", "tool"), (
                    f"Tool message at index {idx} preceded by {prev.get('role')}"
                )
                if prev.get("role") == "assistant":
                    tc_ids = [tc.get("id") for tc in prev.get("tool_calls", [])]
                    assert msg.get("tool_call_id") in tc_ids, (
                        f"Tool message call_id {msg.get('tool_call_id')} not in caller {tc_ids}"
                    )

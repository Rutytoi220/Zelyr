"""Test suite for Adaptive Context Window Budgeting and Historical Tool Compaction.

Covers:
1. Model Context Catalog and context window limit discovery across model families.
2. Token budget calculations and generation headroom reservations.
3. Stage 1 tool output compaction: truncates historical tool results while preserving recent tool results (<2 turns).
4. Stage 1 JSON tool output structure preservation (status, keys, exit codes intact).
5. Stage 2 sliding-window turn pruning and semantic summary anchor injection.
6. Inviolate anchors: root system prompt and immediate user query strictly preserved under pressure.
7. Zero-overhead fast path for short conversations (<50% capacity, sub-millisecond execution).
8. Multi-turn ReAct integration simulation (15 turns, strict budget ceiling enforcement).
9. Facade import parity across zelyr.engine.budget_manager and zelyr.agents.context_budgeter.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import pytest

from zelyr.agents.context_budgeter import (
    AdaptiveContextBudgeter,
    ContextBudgetEngine,
    MODEL_CONTEXT_CATALOG,
    count_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    get_model_context_limit,
    is_tool_result_message,
)
from zelyr.agents.native_orchestrator import NativeOrchestrator


class TestModelContextCatalogAndBudget:
    """Tests for model catalog discovery and token budget calculations."""

    def test_catalog_context_limits_across_families(self):
        """Verify model catalog resolves correct context window capacities."""
        # Qwen
        assert get_model_context_limit("qwen2.5-coder:7b") == 32768
        assert get_model_context_limit("ollama/qwen2.5:32b") == 32768
        assert get_model_context_limit("qwen3:0.6b") == 32768

        # Llama
        assert get_model_context_limit("llama-3.1:8b") == 131072
        assert get_model_context_limit("llama3.2:3b") == 131072
        assert get_model_context_limit("llama-3:8b") == 8192
        assert get_model_context_limit("llama-2:7b") == 4096

        # Frontier cloud models
        assert get_model_context_limit("gpt-4o") == 128000
        assert get_model_context_limit("gpt-4o-mini") == 128000
        assert get_model_context_limit("claude-3-5-sonnet") == 200000
        assert get_model_context_limit("claude-3-5-haiku") == 200000
        assert get_model_context_limit("gemini-1.5-pro") == 2097152
        assert get_model_context_limit("deepseek-r1:14b") == 65536

        # Unknown models default safely to 8192
        assert get_model_context_limit("my-custom-finetune") == 8192
        assert get_model_context_limit(None) == 8192

    def test_budget_calculator_and_reserve(self):
        """Verify reserve margin calculation and target budget headroom."""
        engine = ContextBudgetEngine(reserve_ratio=0.20, min_reserve_tokens=1000, max_reserve_tokens=4096)

        # 8k model: 20% is 1638 tokens reserve -> target budget 6554
        max_tok, reserve = engine.calculate_budget(model_name="llama-3")
        assert max_tok == 8192
        assert reserve == 1638

        # 32k model: 20% is 6553 -> clamped to max_reserve_tokens 4096 -> target budget 28672
        max_tok, reserve = engine.calculate_budget(model_name="qwen2.5-coder:7b")
        assert max_tok == 32768
        assert reserve == 4096

        # 4k model: min_reserve is 1000 -> target budget 3096
        max_tok, reserve = engine.calculate_budget(model_name="llama-2")
        assert max_tok == 4096
        assert reserve == 1000

        # Explicit overrides respected
        max_tok, reserve = engine.calculate_budget(max_context_tokens=16000, reserve_tokens=2500)
        assert max_tok == 16000
        assert reserve == 2500

    def test_context_headroom_telemetry(self):
        """Verify headroom calculation dictionary."""
        engine = ContextBudgetEngine()
        messages = [
            {"role": "system", "content": "System directives"},
            {"role": "user", "content": "Hello model"},
        ]
        headroom = engine.get_context_headroom(messages, model_name="qwen2.5-coder:7b")
        assert headroom["max_context_tokens"] == 32768
        assert headroom["reserve_tokens"] == 4096
        assert headroom["target_budget"] == 28672
        assert headroom["current_tokens"] > 0
        assert headroom["headroom_tokens"] < 28672
        assert headroom["is_over_budget"] is False
        assert 0.0 < headroom["utilization"] < 0.05


class TestStage1ToolCompaction:
    """Tests for Stage 1 historical tool result compaction."""

    def test_historical_tool_compaction_vs_recent_preservation(self):
        """Assert historical tool results (>2 turns old) are compacted while recent ones remain untouched."""
        engine = ContextBudgetEngine(tool_output_token_limit=200, recent_turns_to_preserve=2)

        bulky_tool_output = "Line content: " + ("data_chunk " * 400)  # ~500 tokens
        assert estimate_tokens(bulky_tool_output) > 200

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are Zelyr."},
            # Turn 1 (Historical)
            {"role": "user", "content": "Run tool 1"},
            {"role": "assistant", "content": "Running tool 1", "tool_calls": [{"name": "shell"}]},
            {"role": "tool", "name": "shell", "content": bulky_tool_output},
            # Turn 2 (Historical)
            {"role": "user", "content": "Run tool 2"},
            {"role": "assistant", "content": "Running tool 2", "tool_calls": [{"name": "file_read"}]},
            {"role": "tool", "name": "file_read", "content": bulky_tool_output},
            # Turn 3 (Recent: 2nd most recent user turn)
            {"role": "user", "content": "Run tool 3"},
            {"role": "assistant", "content": "Running tool 3", "tool_calls": [{"name": "inspect"}]},
            {"role": "tool", "name": "inspect", "content": bulky_tool_output},
            # Turn 4 (Recent: latest user turn)
            {"role": "user", "content": "What is the final summary?"},
        ]

        compacted, count = engine.stage1_compact_tool_results(messages, recent_turns_to_preserve=2, tool_output_token_limit=200)

        # Turns 1 and 2 tools should be compacted (2 compacted)
        assert count == 2

        # Turn 1 tool result compacted
        t1_tool = compacted[3]
        assert t1_tool["role"] == "tool"
        assert "[... output truncated:" in t1_tool["content"]
        assert "chars omitted ...]" in t1_tool["content"]
        assert estimate_tokens(t1_tool["content"]) < 200

        # Turn 2 tool result compacted
        t2_tool = compacted[6]
        assert t2_tool["role"] == "tool"
        assert "[... output truncated:" in t2_tool["content"]

        # Turn 3 tool result (recent window) must be completely UNTOUCHED
        t3_tool = compacted[9]
        assert t3_tool["role"] == "tool"
        assert t3_tool["content"] == bulky_tool_output
        assert "[... output truncated:" not in t3_tool["content"]

    def test_single_turn_react_loop_tool_compaction(self):
        """In a deep ReAct loop within a single user query, protect the last 2 tool calls and compact earlier ones."""
        engine = ContextBudgetEngine(tool_output_token_limit=200, recent_turns_to_preserve=2)
        bulky_payload = "Step Output:\n" + ("trace_data_element " * 350)

        # 1 user query, followed by 4 sequential tool iterations
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are Zelyr."},
            {"role": "user", "content": "Diagnose complex failure across 4 components."},
            # Step 1
            {"role": "assistant", "content": "Checking component 1", "tool_calls": [{"name": "comp1"}]},
            {"role": "tool", "name": "comp1", "content": bulky_payload},
            # Step 2
            {"role": "assistant", "content": "Checking component 2", "tool_calls": [{"name": "comp2"}]},
            {"role": "tool", "name": "comp2", "content": bulky_payload},
            # Step 3
            {"role": "assistant", "content": "Checking component 3", "tool_calls": [{"name": "comp3"}]},
            {"role": "tool", "name": "comp3", "content": bulky_payload},
            # Step 4
            {"role": "assistant", "content": "Checking component 4", "tool_calls": [{"name": "comp4"}]},
            {"role": "tool", "name": "comp4", "content": bulky_payload},
        ]

        compacted, count = engine.stage1_compact_tool_results(messages, recent_turns_to_preserve=2, tool_output_token_limit=200)

        # 4 tool calls total; last 2 (steps 3 and 4) are protected, earlier 2 (steps 1 and 2) are compacted
        assert count == 2

        # Step 1 & 2 compacted
        assert "[... output truncated:" in compacted[3]["content"]
        assert "[... output truncated:" in compacted[5]["content"]

        # Step 3 & 4 protected
        assert "[... output truncated:" not in compacted[7]["content"]
        assert "[... output truncated:" not in compacted[9]["content"]

    def test_json_tool_output_structure_preservation(self):
        """Assert JSON tool outputs retain envelope keys and status fields when compacted."""
        engine = ContextBudgetEngine(tool_output_token_limit=200)

        large_stdout = "Stdout log line:\n" + ("output_stream_event " * 500)
        json_tool_content = json.dumps({
            "success": True,
            "exit_code": 0,
            "stdout": large_stdout,
            "command": "git log",
        })

        compacted_text = engine.compact_tool_output(json_tool_content, token_limit=200)

        # Output should be valid JSON
        parsed = json.loads(compacted_text)
        assert parsed["success"] is True
        assert parsed["exit_code"] == 0
        assert parsed["command"] == "git log"
        assert "[... output truncated:" in parsed["stdout"]

    def test_short_tool_results_remain_unmodified(self):
        """Short tool results below token limit should never be truncated."""
        engine = ContextBudgetEngine(tool_output_token_limit=250)
        short_output = "Files: main.py, test.py, config.py (exit code 0)"

        result = engine.compact_tool_output(short_output, token_limit=250)
        assert result == short_output
        assert "[... output truncated:" not in result


class TestStage2TurnPruningAndSummarization:
    """Tests for Stage 2 turn eviction and semantic preamble summarization."""

    def test_stage2_turn_pruning_under_budget_pressure(self):
        """Assert oldest turns are pruned under budget pressure and summary anchor is injected."""
        engine = ContextBudgetEngine()

        system_prompt = {"role": "system", "content": "Base instructions and persona directives."}
        turn_1_user = {"role": "user", "content": "Set up database schema and user permissions."}
        turn_1_assist = {"role": "assistant", "content": "Database schema created with admin tables."}

        turn_2_user = {"role": "user", "content": "Insert test seed data: " + ("record " * 200)}
        turn_2_assist = {"role": "assistant", "content": "Seeded 200 records: " + ("ok " * 200)}

        turn_3_user = {"role": "user", "content": "Run validation tests: " + ("check " * 200)}
        turn_3_assist = {"role": "assistant", "content": "Validation tests passed: " + ("pass " * 200)}

        turn_4_user = {"role": "user", "content": "Latest query: show me user count."}

        messages = [
            system_prompt,
            turn_1_user,
            turn_1_assist,
            turn_2_user,
            turn_2_assist,
            turn_3_user,
            turn_3_assist,
            turn_4_user,
        ]

        total_tokens_before = estimate_messages_tokens(messages)
        assert total_tokens_before > 1200

        # Enforce tight budget of 800 tokens
        pruned_msgs, evicted_count = engine.stage2_prune_turns(messages, target_budget=800)

        assert evicted_count > 0
        total_tokens_after = estimate_messages_tokens(pruned_msgs)
        assert total_tokens_after <= 800

        # Invariant 1: System prompt is preserved at index 0
        assert pruned_msgs[0] == system_prompt

        # Invariant 2: Summary anchor is injected immediately after system prompt
        assert pruned_msgs[1]["role"] == "system"
        assert "[Previous conversation summary:" in pruned_msgs[1]["content"]

        # Invariant 3: Immediate user query (latest turn) is strictly preserved
        assert pruned_msgs[-1] == turn_4_user

        # Invariant 4: Oldest bulky middle turns were pruned
        assert turn_2_user not in pruned_msgs
        assert turn_2_assist not in pruned_msgs

    def test_semantic_summary_content_fidelity(self):
        """Assert summary extracts user questions and tool/assistant outcomes accurately."""
        engine = ContextBudgetEngine()

        evicted_turns = [
            {"role": "user", "content": "Find all Python files in the repository."},
            {"role": "assistant", "content": "Found 42 files.", "tool_calls": [{"name": "file_search"}]},
            {"role": "tool", "name": "file_search", "content": "main.py\nrepl.py"},
            {"role": "user", "content": "Verify linting compliance."},
            {"role": "assistant", "content": "Ruff audit clean with zero errors."},
        ]

        summary = engine.summarize_turns(evicted_turns)

        assert "User asked:" in summary
        assert "Find all Python files" in summary
        assert "Verify linting" in summary
        assert "Tools executed:" in summary
        assert "file_search" in summary
        assert "Key outcomes:" in summary
        assert "Ruff audit clean" in summary


class TestZeroOverheadFastPath:
    """Tests for zero-allocation fast path for short conversations."""

    def test_fast_path_latency_and_identity(self):
        """Short conversations well under 50% budget capacity must return instantly with zero overhead."""
        engine = ContextBudgetEngine()

        messages = [
            {"role": "system", "content": "Directives."},
            {"role": "user", "content": "What time is it?"},
        ]

        # Warm up
        engine.apply_context_budget(messages, max_context_tokens=8192, reserve_tokens=1000)

        # Benchmark 100 runs
        t0 = time.perf_counter()
        for _ in range(100):
            res = engine.apply_context_budget(messages, max_context_tokens=8192, reserve_tokens=1000)
            assert res is messages  # Exact same object reference returned!
        elapsed = time.perf_counter() - t0

        avg_latency_ms = (elapsed / 100) * 1000
        # Must execute in well under 0.1ms (<100 microseconds)
        assert avg_latency_ms < 0.1, f"Fast path too slow: {avg_latency_ms:.4f}ms"


class TestMultiTurnReActIntegration:
    """Tests for multi-turn ReAct simulation maintaining budget compliance across 15 turns."""

    def test_fifteen_turn_react_simulation_within_budget(self):
        """Simulate a 15-turn ReAct conversation accumulating verbose shell and file outputs."""
        orchestrator = NativeOrchestrator(max_context_tokens=4096)
        engine = orchestrator.budget_engine

        # Budget is 4096 - 1000 = 3096 tokens
        budget = 3096

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are Zelyr's autonomous ReAct agent."},
        ]

        # Simulate 15 turns with verbose tool outputs
        for turn_idx in range(1, 16):
            messages.append({"role": "user", "content": f"Turn {turn_idx}: Execute task step {turn_idx}."})
            tool_name = "execute_command" if turn_idx % 2 == 0 else "manage_workspace_file"
            tool_output = f"Result of {tool_name} turn {turn_idx}: " + ("output_line_payload " * 80)

            messages.append({
                "role": "assistant",
                "content": f"Step {turn_idx} in progress.",
                "tool_calls": [{"name": tool_name}],
            })
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": tool_output,
            })
            messages.append({
                "role": "assistant",
                "content": f"Step {turn_idx} completed successfully.",
            })

            # Apply context budget at end of every turn
            compacted = orchestrator.prune_messages(messages, max_context_tokens=4096, reserve_tokens=1000)

            total_tokens = estimate_messages_tokens(compacted)
            assert total_tokens <= budget, (
                f"Turn {turn_idx} exceeded budget: {total_tokens} > {budget}"
            )

            # System prompt is always preserved
            assert compacted[0]["role"] == "system"
            assert compacted[0]["content"] == "You are Zelyr's autonomous ReAct agent."

            # Most recent non-system turn is always preserved
            last_msg = compacted[-1]
            assert "completed successfully" in last_msg["content"] or last_msg["role"] in ("assistant", "user")

        # After 15 turns, verify that summary anchor was created
        summary_msgs = [m for m in compacted if "[Previous conversation summary:" in str(m.get("content", ""))]
        assert len(summary_msgs) == 1


class TestFacadeImportsAndParity:
    """Verify facade re-exports and API parity."""

    def test_facade_symbols(self):
        """All context budget symbols should be imported equivalently from engine and agents."""
        import zelyr.agents.context_budgeter as cb
        import zelyr.engine.budget_manager as bm

        assert cb.ContextBudgetEngine is bm.ContextBudgetEngine
        assert cb.AdaptiveContextBudgeter is bm.AdaptiveContextBudgeter
        assert cb.MODEL_CONTEXT_CATALOG is bm.MODEL_CONTEXT_CATALOG
        assert cb.get_model_context_limit is bm.get_model_context_limit
        assert cb.estimate_tokens is bm.estimate_tokens
        assert cb.count_message_tokens is bm.count_message_tokens
        assert cb.estimate_messages_tokens is bm.estimate_messages_tokens


class TestStage15ThinkingCompaction:
    """Ensure large <think> reasoning blocks are compacted before Stage 2 evicts actual tool observations."""

    def test_stage15_compacts_think_blocks_before_evicting_tool_turns(self):
        """When <think> tags push messages over budget, Stage 1.5 strips <think> so Turn 1 tool observations survive."""
        engine = ContextBudgetEngine()
        huge_think = "<think>\n" + ("Detailed chain of thought reasoning token segment. " * 120) + "\n</think>\n"

        messages = [
            {"role": "system", "content": "You are Zelyr. Keep answers concise."},
            {"role": "user", "content": "Read inspect_target.txt and overwrite it with UPDATED_909."},
            {
                "role": "assistant",
                "content": huge_think,
                "tool_calls": [{"id": "c_read", "type": "function", "function": {"name": "manage_workspace_file", "arguments": '{"action": "read"}'}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c_read",
                "name": "manage_workspace_file",
                "content": "[TOOL OBSERVATION: manage_workspace_file]\nOLD_STALE_VALUE",
            },
            {
                "role": "user",
                "content": "Continue with the write step.",
            },
        ]

        total_before = estimate_messages_tokens(messages)
        # Budget chosen so raw <think> overflows target_budget, while compacted conversation fits easily
        max_ctx = 1600
        reserve = 400
        target_budget = max_ctx - reserve
        assert total_before > target_budget

        compacted = engine.apply_context_budget(messages, max_context_tokens=max_ctx, reserve_tokens=reserve)
        assert estimate_messages_tokens(compacted) <= target_budget

        # Tool observation from Turn 1 must NOT be evicted into a summary anchor!
        assert not any("[Previous conversation summary:" in str(m.get("content", "")) for m in compacted)
        tool_msgs = [m for m in compacted if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "OLD_STALE_VALUE" in tool_msgs[0]["content"]
        # Tool calls metadata intact while bulky <think> was stripped
        assist_msgs = [m for m in compacted if m.get("role") == "assistant"]
        assert len(assist_msgs) == 1
        assert "<think>" not in str(assist_msgs[0].get("content", ""))
        assert assist_msgs[0].get("tool_calls")

    def test_under_budget_preserves_think_blocks(self):
        """Under budget / fast-path conversations must continue preserving <think> tags."""
        engine = ContextBudgetEngine()
        messages = [
            {"role": "system", "content": "Directives."},
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "<think>Short reasoning.</think>\nHi there!"},
            {"role": "user", "content": "Next."},
        ]
        res = engine.apply_context_budget(messages, max_context_tokens=8192, reserve_tokens=1000)
        assert "<think>Short reasoning.</think>" in res[2]["content"]


class TestConfigBudgetLimitsAndFailurePreservation:
    """Verify Commit 5 budgeting source-of-truth, estimates, error retention, and overflow errors."""

    def test_config_source_of_truth_and_heuristic_estimate_labels(self):
        from zelyr.config import ZelyrConfig

        engine = ContextBudgetEngine(reserve_ratio=0.20, min_reserve_tokens=500)
        cfg = ZelyrConfig(ollama_num_ctx=4096)

        # Config supplies effective num_ctx instead of defaulting silently to 32k catalog
        limit, reserve = engine.calculate_budget(model_name="qwen2.5:7b", config=cfg)
        assert limit == 4096
        assert reserve == 819

        # Explicit caller override wins over config
        ov_limit, _ = engine.calculate_budget(model_name="qwen2.5:7b", max_context_tokens=16384, config=cfg)
        assert ov_limit == 16384

        tools = [{"type": "function", "function": {"name": "execute_command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
        headroom = engine.get_context_headroom(
            [{"role": "system", "content": "System"}, {"role": "user", "content": "Hi"}],
            model_name="qwen2.5:7b",
            tools=tools,
            config=cfg,
        )
        assert headroom["token_counting_mode"] == "heuristic_estimate"
        assert headroom["estimated_tools_tokens"] > 0
        assert headroom["estimated_current_tokens"] == headroom["current_tokens"]
        assert headroom["target_budget"] == 4096 - 819 - headroom["estimated_tools_tokens"]

    def test_stage2_preserves_failure_evidence_and_tool_call_arguments(self):
        engine = ContextBudgetEngine()
        messages = [
            {"role": "system", "content": "System instructions."},
            {"role": "user", "content": "Root task: fix and verify config.json."},
            {
                "role": "assistant",
                "content": "Writing file.",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "manage_workspace_file", "arguments": {"path": "config.json"}}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "manage_workspace_file",
                "content": (
                    "[TOOL OBSERVATION: manage_workspace_file]\n"
                    "- Status: SUCCESS\n"
                    "[Outcome Verification Failed]:\n"
                    "- Failed Criterion: file_contains:config.json:READY\n"
                    "- Verifier Reason: Missing READY token\n"
                    + ("filler " * 220)
                ),
            },
            {"role": "user", "content": "Current turn: finalize cleanly."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "verify_outcome", "arguments": {"check": "file_contains", "target": "config.json", "expected": "READY"}}}],
            },
            {"role": "tool", "tool_call_id": "c2", "name": "verify_outcome", "content": "Pending"},
        ]
        pruned, evicted = engine.stage2_prune_turns(messages, target_budget=400)
        assert evicted > 0
        summary_msg = next(m for m in pruned if "[Previous conversation summary:" in str(m.get("content", "")))
        assert "Preserved failures/verification errors" in summary_msg["content"]
        assert "file_contains:config.json:READY" in summary_msg["content"]
        active_assist = next(m for m in pruned if m.get("role") == "assistant" and m.get("tool_calls") and m["tool_calls"][0]["id"] == "c2")
        assert active_assist["tool_calls"][0]["function"]["arguments"] == {"check": "file_contains", "target": "config.json", "expected": "READY"}

    def test_oversized_required_context_fails_explicitly(self):
        from zelyr.engine.budget_manager import ContextBudgetLimitError

        engine = ContextBudgetEngine()
        oversized = [
            {"role": "system", "content": "Mandatory system directive " * 80},
            {"role": "user", "content": "Mandatory root prompt " * 80},
        ]
        with pytest.raises(ContextBudgetLimitError) as exc_info:
            engine.enforce_context_budget(oversized, max_context_tokens=500, reserve_tokens=100)
        err_dict = exc_info.value.to_dict()
        assert err_dict["error_type"] == "CONTEXT_BUDGET_EXCEEDED"
        assert err_dict["required_tokens"] > err_dict["target_budget"]

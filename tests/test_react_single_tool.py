"""Verification Test Suite: Autonomous Multi-Step ReAct Loop & Single Tool Dispatch.

Tests:
1. System prompt invariants (CRITICAL TOOL CALLING RULES).
2. NativeOrchestrator.extract_first_tool_call() tolerant parsing on concatenated JSON,
   code fences, list formats, and thinking tags.
3. Strict single-tool dispatch constraint: model dumping concatenated JSON only executes the first call.
4. Autonomous multi-step ReAct loop: observations fed back to the model without user intervention.
5. Max step limit bound (5 steps).
"""

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, patch

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.core.system_prompt import (
    HARDENED_SYSTEM_DIRECTIVES,
    SOM_REACT_SYSTEM_PROMPT,
    extract_xml_tool_call,
)


def test_system_prompt_rules():
    print("[CHECK 1] Testing System Prompt Tool Calling Rules...")
    for prompt_text, name in [
        (HARDENED_SYSTEM_DIRECTIVES, "HARDENED_SYSTEM_DIRECTIVES"),
        (SOM_REACT_SYSTEM_PROMPT, "SOM_REACT_SYSTEM_PROMPT"),
    ]:
        assert "CRITICAL TOOL CALLING RULES:" in prompt_text, f"Missing rules section in {name}"
        assert "Output EXACTLY ONE tool call per response" in prompt_text, f"Missing single tool rule in {name}"
        assert "NEVER concatenate multiple JSON tool calls" in prompt_text, f"Missing anti-concatenation rule in {name}"
        assert "get_page_snapshot" in prompt_text, f"Missing get_page_snapshot directive in {name}"
    print("✓ Check 1 PASSED: System prompts contain explicit single tool and multi-step constraints.\n")


def test_tolerant_json_parsing():
    print("[CHECK 2] Testing Tolerant JSON Tool Call Extraction...")
    orch = NativeOrchestrator()
    known = {"interact_with_browser", "manage_desktop_window", "interact_with_ui"}

    # 1. Direct concatenated JSON (the Qwen-7B defect)
    concatenated = (
        '{"name": "interact_with_browser", "arguments": {"action": "open_tab", "url": "https://google.com"}}'
        '{"name": "interact_with_browser", "arguments": {"action": "click_element", "element_id": 1}}'
    )
    res = orch.extract_first_tool_call(concatenated, known)
    assert res is not None, "Failed to extract from concatenated JSON"
    assert res["name"] == "interact_with_browser"
    assert res["arguments"]["action"] == "open_tab"
    assert res["arguments"]["url"] == "https://google.com"

    # 2. Concatenated JSON separated by newline
    newline_concat = (
        '{"name": "manage_desktop_window", "arguments": {"action": "get_active"}}\n'
        '{"name": "interact_with_browser", "arguments": {"action": "click"}}'
    )
    res = orch.extract_first_tool_call(newline_concat, known)
    assert res is not None
    assert res["name"] == "manage_desktop_window"
    assert res["arguments"]["action"] == "get_active"

    # 3. Code-fenced concatenated JSON
    fenced = (
        "```json\n"
        '{"name": "interact_with_browser", "arguments": {"action": "get_page_snapshot"}}\n'
        '{"name": "interact_with_browser", "arguments": {"action": "fill_element", "element_id": 2, "text": "zelyr"}}\n'
        "```"
    )
    res = orch.extract_first_tool_call(fenced, known)
    assert res is not None
    assert res["name"] == "interact_with_browser"
    assert res["arguments"]["action"] == "get_page_snapshot"

    # 4. Thinking tags before concatenated JSON
    thinking = (
        "<think>\nI need to open the tab first and then click.\n</think>\n"
        '{"name": "interact_with_browser", "arguments": {"action": "open_tab", "url": "https://zelyr.ai"}}\n'
        '{"name": "interact_with_browser", "arguments": {"action": "click_element", "element_id": 5}}'
    )
    res = orch.extract_first_tool_call(thinking, known)
    assert res is not None
    assert res["name"] == "interact_with_browser"
    assert res["arguments"]["action"] == "open_tab"

    # 5. JSON list format
    list_format = (
        '[{"name": "interact_with_browser", "arguments": {"action": "open_tab", "url": "https://zelyr.ai"}}, '
        '{"name": "interact_with_browser", "arguments": {"action": "click_element"}}]'
    )
    res = orch.extract_first_tool_call(list_format, known)
    assert res is not None
    assert res["name"] == "interact_with_browser"
    assert res["arguments"]["action"] == "open_tab"

    # 6. String-encoded arguments
    string_args = (
        '{"name": "interact_with_browser", "arguments": "{\\"action\\": \\"open_tab\\", \\"url\\": \\"https://test.com\\"}"}'
    )
    res = orch.extract_first_tool_call(string_args, known)
    assert res is not None
    assert res["name"] == "interact_with_browser"
    assert res["arguments"]["action"] == "open_tab"

    # 7. XML tag with concatenated JSON inside
    xml_concat = (
        '<tool_call>{"name": "interact_with_browser", "arguments": {"action": "switch_tab", "query": "gemini"}}'
        '{"name": "interact_with_browser", "arguments": {"action": "click"}}</tool_call>'
    )
    res = extract_xml_tool_call(xml_concat)
    assert res is not None
    assert res["function"]["name"] == "interact_with_browser"
    assert res["function"]["arguments"]["action"] == "switch_tab"

    # 8. Regular conversational text returns None
    assert orch.extract_first_tool_call("I will help you switch tabs now.", known) is None
    assert orch.extract_first_tool_call("", known) is None

    print("✓ Check 2 PASSED: extract_first_tool_call handles concatenated JSON and edge cases perfectly.\n")


def test_autonomous_react_loop_and_single_dispatch():
    print("[CHECK 3] Testing Autonomous Multi-Step ReAct Loop & Single Dispatch...")
    orch = NativeOrchestrator()

    executed_tools = []

    async def mock_execute(name, **kwargs):
        executed_tools.append((name, kwargs))
        if kwargs.get("action") == "open_tab":
            return {"status": "success", "action": "open_tab", "tab_id": 101}
        elif kwargs.get("action") == "get_page_snapshot":
            return {"elements": [{"element_id": 1, "tag": "input", "role": "searchbox"}]}
        elif kwargs.get("action") == "fill_element":
            return {"status": "success", "action": "fill_element", "element_id": 1}
        return {"status": "ok"}

    turn_counter = 0

    @contextlib.asynccontextmanager
    async def mock_stream_client(method, url, **kwargs):
        nonlocal turn_counter
        turn_counter += 1
        resp = AsyncMock()
        resp.status_code = 200

        # Turn 1: Model outputs CONCATENATED JSON (open_tab + premature fill_element)
        if turn_counter == 1:
            lines = [
                'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"interact_with_browser\\", \\"arguments\\": {\\"action\\": \\"open_tab\\", \\"url\\": \\"https://google.com\\"}}"}}]}',
                'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"interact_with_browser\\", \\"arguments\\": {\\"action\\": \\"fill_element\\", \\"element_id\\": 99, \\"text\\": \\"premature\\"}}"}}]}',
                "data: [DONE]",
            ]
        # Turn 2: Observation received -> Model requests page snapshot
        elif turn_counter == 2:
            lines = [
                'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"interact_with_browser\\", \\"arguments\\": {\\"action\\": \\"get_page_snapshot\\"}}"}}]}',
                "data: [DONE]",
            ]
        # Turn 3: Snapshot received -> Model fills element 1
        elif turn_counter == 3:
            lines = [
                'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"interact_with_browser\\", \\"arguments\\": {\\"action\\": \\"fill_element\\", \\"element_id\\": 1, \\"text\\": \\"zelyr search\\"}}"}}]}',
                "data: [DONE]",
            ]
        # Turn 4: Completion text
        else:
            lines = [
                'data: {"choices": [{"delta": {"content": "I have successfully opened Google, inspected elements, and entered your query."}}]}',
                "data: [DONE]",
            ]

        async def aiter_lines():
            for l in lines:
                yield l

        resp.aiter_lines = aiter_lines
        yield resp

    async def run_react_test():
        payload = {
            "messages": [{"role": "user", "content": "Open Google and search for zelyr"}],
        }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None), \
             patch("httpx.AsyncClient") as mock_client:

            mock_client.return_value.__aenter__.return_value.stream = mock_stream_client
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            emitted_texts = []
            emitted_tools = []

            async for chunk in orch.generate_stream(payload):
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        emitted_texts.append(delta["content"])
                    if "tool_calls" in delta:
                        emitted_tools.extend(delta["tool_calls"])

            final_text = "".join(emitted_texts)

            # Assert execution count and sequence
            assert len(executed_tools) == 3, f"Expected exactly 3 tool executions, got {len(executed_tools)}"

            # Step 1: open_tab executed; premature fill_element was NOT executed
            assert executed_tools[0][1]["action"] == "open_tab"
            assert executed_tools[0][1]["url"] == "https://google.com"

            # Step 2: get_page_snapshot executed
            assert executed_tools[1][1]["action"] == "get_page_snapshot"

            # Step 3: fill_element executed with element_id=1 (not the premature element_id=99)
            assert executed_tools[2][1]["action"] == "fill_element"
            assert executed_tools[2][1]["element_id"] == 1

            # Step 4: Final text cleanly streamed without raw JSON leakage
            assert "I have successfully opened Google" in final_text
            assert '{"name":' not in final_text, "Raw JSON tool call leaked into streamed content!"

            # Assert messages history inside payload contains tool observations
            tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
            assert len(tool_messages) == 3, f"Expected 3 tool observation messages in history, got {len(tool_messages)}"
            for tm in tool_messages:
                assert tm["role"] == "tool"
                assert "content" in tm

    asyncio.run(run_react_test())
    print("✓ Check 3 PASSED: Autonomous ReAct loop cleanly executed 3 steps and suppressed raw JSON.\n")


def test_max_depth_bound():
    print("[CHECK 4] Testing ReAct Max Depth Bound (5 steps)...")
    orch = NativeOrchestrator()
    loop_count = 0

    async def mock_execute(name, **kwargs):
        return {"status": "looping"}

    @contextlib.asynccontextmanager
    async def mock_infinite_loop_stream(method, url, **kwargs):
        nonlocal loop_count
        loop_count += 1
        resp = AsyncMock()
        resp.status_code = 200

        lines = [
            'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"manage_desktop_window\\", \\"arguments\\": {\\"action\\": \\"get_active\\"}}"}}]}',
            "data: [DONE]",
        ]

        async def aiter_lines():
            for l in lines:
                yield l

        resp.aiter_lines = aiter_lines
        yield resp

    async def run_depth_test():
        payload = {
            "messages": [{"role": "user", "content": "Keep looping"}],
        }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None), \
             patch("httpx.AsyncClient") as mock_client:

            mock_client.return_value.__aenter__.return_value.stream = mock_infinite_loop_stream
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            streamed = []
            async for chunk in orch.generate_stream(payload):
                choices = chunk.get("choices", [])
                if choices:
                    c = choices[0].get("delta", {}).get("content", "")
                    if c:
                        streamed.append(c)

            out = "".join(streamed)
            assert "Reached maximum autonomous ReAct steps (5)" in out, f"Expected max steps notice, got: {out}"
            assert loop_count == 5, f"Expected exactly 5 LLM requests before cutoff, got {loop_count}"

    asyncio.run(run_depth_test())
    print("✓ Check 4 PASSED: ReAct loop bounded strictly at 5 steps.\n")


def test_native_api_multiple_tool_calls_enforcement():
    print("[CHECK 5] Testing Native API Multi-Tool Call Suppression (strictly one dispatch)...")
    orch = NativeOrchestrator()
    executed_tools = []

    async def mock_execute(name, **kwargs):
        executed_tools.append(name)
        return {"status": "ok"}

    @contextlib.asynccontextmanager
    async def mock_multi_tc_stream(method, url, **kwargs):
        resp = AsyncMock()
        resp.status_code = 200

        # API returns TWO tool calls in choices[0].delta
        lines = [
            'data: {"choices": [{"delta": {"tool_calls": ['
            '{"index": 0, "id": "call_1", "type": "function", "function": {"name": "interact_with_browser", "arguments": "{\\"action\\": \\"switch_tab\\", \\"query\\": \\"gemini\\"}"}}, '
            '{"index": 1, "id": "call_2", "type": "function", "function": {"name": "manage_desktop_window", "arguments": "{\\"action\\": \\"get_active\\"}"}}'
            ']}}]}',
            "data: [DONE]",
        ]

        async def aiter_lines():
            for l in lines:
                yield l

        resp.aiter_lines = aiter_lines
        yield resp

    async def run_multi_tc_test():
        payload = {
            "messages": [{"role": "user", "content": "Switch tab and get active window"}],
        }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None), \
             patch("httpx.AsyncClient") as mock_client:

            mock_client.return_value.__aenter__.return_value.stream = mock_multi_tc_stream
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            # We test only depth 0
            async for chunk in orch.generate_stream(payload, depth=4):
                pass

            assert len(executed_tools) == 1, f"Expected strictly 1 tool execution, got {len(executed_tools)}"
            assert executed_tools[0] == "interact_with_browser"

    asyncio.run(run_multi_tc_test())
    print("✓ Check 5 PASSED: Native API multi-tool calls clamped to index 0 strictly.\n")


def test_three_tier_hierarchy_and_guardrails():
    print("[CHECK 6] Testing Three-Tier Routing Hierarchy & Tier 3 Guardrails...")
    from zelyr.tools.browser_cdp import InteractWithBrowserTool
    from zelyr.tools.vision import InteractWithUITool

    # 1. Check prompt hierarchy section
    for prompt_text, name in [
        (HARDENED_SYSTEM_DIRECTIVES, "HARDENED_SYSTEM_DIRECTIVES"),
        (SOM_REACT_SYSTEM_PROMPT, "SOM_REACT_SYSTEM_PROMPT"),
    ]:
        assert "STRICT THREE-TIER TOOL ROUTING HIERARCHY:" in prompt_text, f"Missing routing hierarchy in {name}"
        assert "TIER 1 (OS & CLI - FIRST CHOICE):" in prompt_text, f"Missing Tier 1 in {name}"
        assert "TIER 2 (WEB EXTENSION & DOM - MANDATORY FOR BROWSERS):" in prompt_text, f"Missing Tier 2 in {name}"
        assert "TIER 3 (VISION & MOUSE/KEYBOARD - ABSOLUTE LAST RESORT):" in prompt_text, f"Missing Tier 3 in {name}"
        assert "NEVER use 'interact_with_ui' on web pages or browser tabs." in prompt_text, f"Missing prohibition in {name}"
        assert "If the target is inside a browser, 'interact_with_ui' is STRICTLY FORBIDDEN." in prompt_text, f"Missing prohibition in {name}"

    # 2. Check tool descriptions
    ui_tool = InteractWithUITool()
    browser_tool = InteractWithBrowserTool()

    assert "STRICT RESTRICTION: DO NOT use this tool for web browsers or web pages" in ui_tool.description
    assert "If interacting with a browser tab, you MUST use interact_with_browser with click_element or fill_element." in ui_tool.description
    assert "Use 'click_element' with 'element_id' from the latest get_page_snapshot" in browser_tool.description

    # 3. Check InteractWithUITool execution guardrail against web element queries
    async def run_guardrail_test():
        # element_id passed directly
        res = await ui_tool.execute({"instruction": "click", "element_id": 2})
        assert res.success is False
        assert "STRICT RESTRICTION" in res.error

        # action=click_element passed
        res = await ui_tool.execute({"action": "click_element", "instruction": "click button"})
        assert res.success is False
        assert "STRICT RESTRICTION" in res.error

        # instruction mentioning snapshot element #
        res = await ui_tool.execute({"instruction": "Click element #3 on the webpage"})
        assert res.success is False
        assert "STRICT RESTRICTION" in res.error

    asyncio.run(run_guardrail_test())

    # 4. Check NativeOrchestrator redirection
    orch = NativeOrchestrator()
    executed_tools = []

    async def mock_execute(name, **kwargs):
        executed_tools.append((name, kwargs))
        return {"status": "ok"}

    @contextlib.asynccontextmanager
    async def mock_erroneous_vlm_stream(method, url, **kwargs):
        resp = AsyncMock()
        resp.status_code = 200
        # Model erroneously dispatched interact_with_ui with element_id after snapshot
        lines = [
            'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"interact_with_ui\\", \\"arguments\\": {\\"action\\": \\"click_element\\", \\"element_id\\": 4}}"}}]}',
            "data: [DONE]",
        ]

        async def aiter_lines():
            for l in lines:
                yield l

        resp.aiter_lines = aiter_lines
        yield resp

    async def run_redirect_test():
        payload = {
            "messages": [{"role": "user", "content": "Click element 4 on Google"}],
        }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None), \
             patch("httpx.AsyncClient") as mock_client:

            mock_client.return_value.__aenter__.return_value.stream = mock_erroneous_vlm_stream
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            async for chunk in orch.generate_stream(payload, depth=4):
                pass

            assert len(executed_tools) == 1, f"Expected 1 tool execution, got {len(executed_tools)}"
            assert executed_tools[0][0] == "interact_with_browser", f"Expected redirection to interact_with_browser, got {executed_tools[0][0]}"
            assert executed_tools[0][1]["element_id"] == 4

    asyncio.run(run_redirect_test())
    print("✓ Check 6 PASSED: Hierarchy, tool descriptions, and Tier 3 guardrails validated.\n")


if __name__ == "__main__":
    print("============================================================")
    print("Zelyr AUTONOMOUS MULTI-STEP REACT LOOP & SINGLE TOOL TEST")
    print("============================================================\n")
    test_system_prompt_rules()
    test_tolerant_json_parsing()
    test_autonomous_react_loop_and_single_dispatch()
    test_max_depth_bound()
    test_native_api_multiple_tool_calls_enforcement()
    test_three_tier_hierarchy_and_guardrails()
    print("============================================================")
    print("✅ ALL REACT LOOP & SINGLE TOOL DISPATCH CHECKS PASSED!")
    print("============================================================")

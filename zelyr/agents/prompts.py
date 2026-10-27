def _build_react_prompt(intent: str) -> str:
    return "You are Zelyr. Use a strict ReAct loop (Thought, Action, Observation). Ensure you output tool calls properly. If you are finished, output a <final_answer> marker."

def _build_tool_guidelines() -> str:
    return "Guidelines: 1. Use the provided tools. 2. Handle errors gracefully."

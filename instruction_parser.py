"""Instruction parser and Hybrid Fastest Path Router for Zelyr brain.

Parses LLM output into structured instructions and prioritizes native CLI/API execution
(e.g., xdg-open, systemctl, curl) over slow/brittle GUI mouse and keyboard automation.
"""

from typing import Any, Dict, Optional, Tuple
from brain.parser import parse as _brain_parse
from zelyr.core.fast_path_router import FastestPathRouter, FastPathPlan


def parse(text: str) -> Dict[str, Any]:
    """Parse LLM response and annotate instructions with Fastest Path CLI alternatives.

    If an action (e.g. opening a browser, launching an app, managing a system service)
    can be accomplished via native CLI commands, it is tagged with fast_path=True and
    the exact command list.
    """
    res = _brain_parse(text)
    if not isinstance(res, dict):
        return res

    if res.get("type") == "instruction":
        action = res.get("action", "")
        params = res.get("params", "")
        plan = FastestPathRouter.can_route_to_cli(action, params)
        if plan:
            res["fast_path"] = True
            res["cli_command"] = plan.command
            res["fast_path_description"] = plan.description
            res["fast_path_category"] = plan.category
    elif res.get("type") == "instructions":
        for instr in res.get("instructions", []):
            action = instr.get("action", "")
            params = instr.get("params", "")
            plan = FastestPathRouter.can_route_to_cli(action, params)
            if plan:
                instr["fast_path"] = True
                instr["cli_command"] = plan.command
                instr["fast_path_description"] = plan.description
                instr["fast_path_category"] = plan.category

    return res


def execute_instruction_fast_path(instruction_dict: Dict[str, Any]) -> Tuple[bool, str]:
    """Executes an instruction via native CLI if a fast path exists, or signals fallback."""
    if instruction_dict.get("fast_path") and instruction_dict.get("cli_command"):
        plan = FastPathPlan(
            original_action=instruction_dict.get("action", "unknown"),
            command=instruction_dict["cli_command"],
            description=instruction_dict.get("fast_path_description", ""),
            confidence=0.95,
            category=instruction_dict.get("fast_path_category", "general"),
        )
        return FastestPathRouter.execute_fast_path(plan)

    return False, "No native CLI fast path available. Fallback to physical GUI automation."


__all__ = ["parse", "FastestPathRouter", "FastPathPlan", "execute_instruction_fast_path"]

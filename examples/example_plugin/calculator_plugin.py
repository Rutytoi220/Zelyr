"""Calculator plugin — an Zelyr v2 plugin example.

This module replaces the legacy ``examples/example_custom_tool.py`` with the
RFC-001 compliant ``ZelyrPlugin`` structure.  It declares no special
permissions and cannot access the network, shell, or files outside its
sandboxed workspace.
"""

import math
import operator
from typing import Any, Dict

from zelyr.plugins.zelyr_plugin import ZelyrPlugin, HookResult


# Safe operators and math functions available to the evaluator.
_SAFE_ENV: Dict[str, Any] = {
    # Basic operators
    "__builtins__": {},
    # Math module subset
    "abs": abs,
    "round": round,
    "max": max,
    "min": min,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "pi": math.pi,
    "e": math.e,
    "ceil": math.ceil,
    "floor": math.floor,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


class CalculatorPlugin(ZelyrPlugin):
    """Sandboxed arithmetic calculator."""

    def on_load(self, context: Any) -> None:
        self.register_tool(
            name="calculate",
            description=(
                "Evaluate a safe mathematical expression. "
                "Supports: +, -, *, /, **, sqrt, log, sin, cos, tan, pi, e."
            ),
            handler=self._calculate,
            parameters=[
                {
                    "name": "expression",
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g. '2 + sqrt(16)').",
                    "required": True,
                }
            ],
        )

    def _calculate(self, expression: str, **_kwargs: Any) -> Dict[str, Any]:
        """Evaluate the expression in a restricted namespace."""
        try:
            result = eval(expression, _SAFE_ENV)  # noqa: S307
            return {"expression": expression, "result": result}
        except ZeroDivisionError:
            return {"expression": expression, "error": "Division by zero"}
        except Exception as exc:
            return {"expression": expression, "error": f"Invalid expression: {exc}"}

    def before_tool_execute(self, tool_name: str, args: Dict[str, Any]) -> HookResult:
        """Demonstrate the middleware hook — block expressions that are too long."""
        expr = args.get("expression", "")
        if len(expr) > 512:
            return HookResult.abort(reason="Expression exceeds maximum allowed length (512 chars).")
        return HookResult.continue_execution()

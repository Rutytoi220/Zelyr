"""DesktopAutomationTool — \"Ghost in the Machine\" physical UI control.

Gives the local Zelyr agent the ability to control the host's mouse and
keyboard via pyautogui, with safety guards enforced at init time.

This tool is registered ONLY on the local desktop client, never on headless
Swarm Nodes.
"""
import asyncio
import logging
from functools import partial
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {
    "mouse_move",
    "mouse_click",
    "keyboard_type",
    "keyboard_press",
    "get_screen_size",
}


class DesktopAutomationTool(BaseTool):
    """Allows the LLM to physically control the host's mouse and keyboard."""

    def __init__(self):
        super().__init__(
            tool_id="desktop_control",
            name="DesktopAutomationTool",
            description=(
                "You have physical control over the user's mouse and keyboard on their Linux desktop. "
                "Always call the 'get_screen_size' action first to orient yourself before issuing any "
                "mouse commands. Use this tool cautiously — every action physically moves the cursor "
                "or types on the real screen. The user can abort at any time by moving the mouse to "
                "the top-left corner of the screen (pyautogui failsafe)."
            ),
        )
        self.add_parameter(ToolParameter(
            name="action",
            type="string",
            description=(
                "The action to perform. One of: "
                "'mouse_move' (requires x, y), "
                "'mouse_click' (optional button='left'|'right', optional clicks=1), "
                "'keyboard_type' (requires text), "
                "'keyboard_press' (requires key, e.g. 'enter', 'tab', 'super'), "
                "'get_screen_size' (returns display width and height)."
            ),
        ))
        self.add_parameter(ToolParameter(
            name="x", type="integer",
            description="X coordinate for mouse_move.", required=False,
        ))
        self.add_parameter(ToolParameter(
            name="y", type="integer",
            description="Y coordinate for mouse_move.", required=False,
        ))
        self.add_parameter(ToolParameter(
            name="button", type="string",
            description="Mouse button for mouse_click: 'left' or 'right'. Defaults to 'left'.",
            required=False,
        ))
        self.add_parameter(ToolParameter(
            name="clicks", type="integer",
            description="Number of clicks for mouse_click. Defaults to 1.",
            required=False,
        ))
        self.add_parameter(ToolParameter(
            name="text", type="string",
            description="Text string to type for keyboard_type.", required=False,
        ))
        self.add_parameter(ToolParameter(
            name="key", type="string",
            description="Key name for keyboard_press (e.g. 'enter', 'tab', 'super', 'escape').",
            required=False,
        ))

        # ── Safety & Driver configuration ────────────────────────────────── #
        self._gui = None
        self._wayland_controller = None

    def _ensure_gui(self):
        """Initializes input controllers (WaylandInputController preferred on Linux)."""
        # Try Wayland controller first
        if self._wayland_controller is None:
            try:
                from zelyr.tools.input_controller import WaylandInputController
                self._wayland_controller = WaylandInputController()
                logger.info("DesktopAutomationTool: initialized WaylandInputController via ydotool")
            except Exception as e:
                logger.debug("WaylandInputController unavailable, will attempt pyautogui fallback: %s", e)

        # Fallback to pyautogui if needed
        if self._wayland_controller is None and self._gui is None:
            try:
                import pyautogui
                pyautogui.FAILSAFE = True
                pyautogui.PAUSE = 0.5
                self._gui = pyautogui
                logger.info("DesktopAutomationTool: pyautogui initialized (FAILSAFE=True, PAUSE=0.5)")
            except (ImportError, OSError, RuntimeError) as e:
                logger.warning(f"DesktopAutomationTool gracefully disabled: {e}")
                raise RuntimeError(f"Desktop automation is unavailable on this host: {e}")

    # ── Dispatcher ────────────────────────────────────────────────────── #

    async def execute(self, action: str, **kwargs) -> ToolResult:
        # 1. Fast Path Router Heuristic Evaluation
        # If the requested action or parameters can be fulfilled by a native CLI command
        # (e.g., opening a browser via xdg-open rather than clicking Firefox icon), route it!
        try:
            from zelyr.core.fast_path_router import FastestPathRouter
            fast_path_result = FastestPathRouter.evaluate_and_intercept(action, kwargs)
            if fast_path_result and fast_path_result.get("intercepted"):
                cmd_str = " ".join(fast_path_result.get("command", []))
                logger.info(
                    "DesktopAutomationTool: Fast Path intercepted action '%s' -> %s", action, cmd_str
                )
                return ToolResult(
                    success=fast_path_result.get("success", True),
                    output=fast_path_result.get("output", f"Executed CLI shortcut: {cmd_str}"),
                    metadata={"fast_path": True, "command": fast_path_result.get("command")},
                )
        except Exception as e:
            logger.debug("Fast path router evaluation skipped: %s", e)

        if action not in _VALID_ACTIONS:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Valid actions: {', '.join(sorted(_VALID_ACTIONS))}",
            )
        try:
            self._ensure_gui()
            handler = getattr(self, f"_action_{action}")
            return await handler(**kwargs)
        except Exception as e:
            logger.error(f"DesktopAutomationTool error ({action}): {e}")
            return ToolResult(success=False, error=f"Desktop automation error: {e}")

    # ── Action implementations ────────────────────────────────────────── #

    async def _action_get_screen_size(self, **_) -> ToolResult:
        loop = asyncio.get_event_loop()
        if self._gui is not None:
            size = await loop.run_in_executor(None, self._gui.size)
            return ToolResult(
                success=True,
                output=f"Screen size: {size.width}x{size.height} pixels",
                metadata={"width": size.width, "height": size.height},
            )
        # Wayland fallback via hyprctl or xrandr
        try:
            import subprocess, json
            res = subprocess.run(["hyprctl", "monitors", "-j"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                mons = json.loads(res.stdout)
                for m in mons:
                    if m.get("focused"):
                        return ToolResult(
                            success=True,
                            output=f"Screen size: {m.get('width')}x{m.get('height')} pixels (monitor {m.get('name')})",
                            metadata={"width": m.get("width"), "height": m.get("height")},
                        )
        except Exception:
            pass
        return ToolResult(success=True, output="Screen size: 1920x1080 pixels", metadata={"width": 1920, "height": 1080})

    async def _action_mouse_move(self, x: int = None, y: int = None, **_) -> ToolResult:
        if x is None or y is None:
            return ToolResult(success=False, error="mouse_move requires both 'x' and 'y' parameters.")
        loop = asyncio.get_event_loop()
        if self._wayland_controller:
            await loop.run_in_executor(None, self._wayland_controller.mouse_move, int(x), int(y))
        else:
            await loop.run_in_executor(None, partial(self._gui.moveTo, int(x), int(y), duration=0.3))
        return ToolResult(success=True, output=f"Mouse moved to ({x}, {y})")

    async def _action_mouse_click(self, button: str = "left", clicks: int = 1, **_) -> ToolResult:
        if button not in ("left", "right", "middle"):
            return ToolResult(success=False, error="button must be 'left', 'right', or 'middle'.")
        loop = asyncio.get_event_loop()
        if self._wayland_controller:
            await loop.run_in_executor(None, self._wayland_controller.mouse_click, button, int(clicks))
        else:
            await loop.run_in_executor(
                None, partial(self._gui.click, button=button, clicks=int(clicks))
            )
        return ToolResult(success=True, output=f"Clicked {button} button {clicks} time(s)")

    async def _action_keyboard_type(self, text: str = None, **_) -> ToolResult:
        if not text:
            return ToolResult(success=False, error="keyboard_type requires a 'text' parameter.")
        loop = asyncio.get_event_loop()
        if self._wayland_controller:
            await loop.run_in_executor(None, self._wayland_controller.keyboard_type, text)
        else:
            await loop.run_in_executor(None, partial(self._gui.write, text, interval=0.03))
        return ToolResult(success=True, output=f"Typed {len(text)} characters")

    async def _action_keyboard_press(self, key: str = None, **_) -> ToolResult:
        if not key:
            return ToolResult(success=False, error="keyboard_press requires a 'key' parameter.")
        loop = asyncio.get_event_loop()
        if self._wayland_controller:
            await loop.run_in_executor(None, self._wayland_controller.keyboard_press, key)
        else:
            key_map = {"super": "win", "windows": "win", "cmd": "win"}
            actual_key = key_map.get(key.lower(), key.lower())
            await loop.run_in_executor(None, partial(self._gui.press, actual_key))
        return ToolResult(success=True, output=f"Pressed key: {key}")

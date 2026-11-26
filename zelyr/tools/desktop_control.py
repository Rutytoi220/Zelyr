"""DesktopAutomationTool — Safe, verified Linux desktop UI observation & actuation.

Integrates Hyprland IPC window/monitor geometry, bounded coordinate/key validation,
optional focus verification prior to input dispatch, read-only inspection modes,
and structured error envelopes over Zelyr's existing WaylandInputController and
screen_perception observation pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional

from zelyr.engine.hyprland_ipc import HyprlandIPCClient
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult
from zelyr.tools.input_controller import (
    BackendUnavailableError,
    DesktopUnavailableError,
    InputControlError,
    InputFailedError,
    InputTimeoutError,
    InvalidCoordinatesError,
    InvalidInputKeyError,
    WaylandInputController,
    YdoToolPermissionError,
)
from zelyr.tools.screen_perception import capture_desktop_observation

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {
    "get_screen_size",
    "observe",
    "screenshot",
    "inspect_windows",
    "get_active_window",
    "get_cursor_position",
    "mouse_move",
    "mouse_click",
    "double_click",
    "mouse_drag",
    "mouse_scroll",
    "keyboard_type",
    "keyboard_press",
}

_MUTATING_ACTIONS = {
    "mouse_move",
    "mouse_click",
    "double_click",
    "mouse_drag",
    "mouse_scroll",
    "keyboard_type",
    "keyboard_press",
}


class DesktopAutomationTool(BaseTool):
    """Controls mouse, keyboard, and window-level visual/layout observation on Linux desktops."""

    def __init__(
        self,
        *,
        wayland_controller: Optional[WaylandInputController] = None,
        ipc_client: Optional[HyprlandIPCClient] = None,
    ):
        super().__init__(
            tool_id="desktop_control",
            name="desktop_control",
            description=(
                "Click/double-click/drag/scroll mouse, type text, press keys, or inspect desktop windows. "
                "Supports 'expected_window' focus guard and validated desktop (x, y) coordinates."
            ),
            tier=3,
            is_core=False,
            requires_bridge=False,
            requires_window=True,
            token_cost=350,
        )
        self.add_parameter(ToolParameter(
            name="action",
            type="string",
            description="Desktop input or inspection action to run.",
            enum=[
                "mouse_click",
                "double_click",
                "mouse_move",
                "keyboard_type",
                "keyboard_press",
                "mouse_drag",
                "mouse_scroll",
                "observe",
                "inspect_windows",
                "get_screen_size",
                "get_active_window",
                "get_cursor_position",
                "screenshot",
            ],
            required=True,
        ))
        self.add_parameter(ToolParameter(
            name="x",
            type="integer",
            description="Logical desktop X coordinate for pointer actions.",
            required=False,
        ))
        self.add_parameter(ToolParameter(
            name="y",
            type="integer",
            description="Logical desktop Y coordinate for pointer actions.",
            required=False,
        ))
        self.add_parameter(ToolParameter(
            name="end_x",
            type="integer",
            description="Ending logical desktop X coordinate for mouse_drag.",
            required=False,
        ))
        self.add_parameter(ToolParameter(
            name="end_y",
            type="integer",
            description="Ending logical desktop Y coordinate for mouse_drag.",
            required=False,
        ))
        self.add_parameter(ToolParameter(
            name="button",
            type="string",
            description="Mouse button for mouse_click or mouse_drag.",
            enum=["left", "right", "middle"],
            required=False,
        ))
        self.add_parameter(ToolParameter(
            name="clicks",
            type="integer",
            description="Click count (1-5, default 1).",
            required=False,
        ))
        self.add_parameter(ToolParameter(
            name="direction",
            type="string",
            description="Scroll direction for mouse_scroll.",
            enum=["up", "down", "left", "right"],
            required=False,
        ))
        self.add_parameter(ToolParameter(
            name="amount",
            type="integer",
            description="Scroll wheel tick count (default 3).",
            required=False,
        ))
        self.add_parameter(ToolParameter(
            name="text",
            type="string",
            description="Text string for keyboard_type.",
            required=False,
        ))
        self.add_parameter(ToolParameter(
            name="key",
            type="string",
            description="Key name or combo for keyboard_press (e.g. 'enter', 'tab', 'ctrl+a', 'ctrl+c').",
            required=False,
        ))
        self.add_parameter(ToolParameter(
            name="press_enter",
            type="boolean",
            description="For keyboard_type, also press Enter after typing (default false).",
            required=False,
        ))
        self.add_parameter(ToolParameter(
            name="expected_window",
            type="string",
            description="Optional target window title/class/address that MUST be focused before sending input.",
            required=False,
        ))
        self.add_parameter(ToolParameter(
            name="window",
            type="string",
            description="Optional window target for 'observe'/'screenshot' or expected window verification.",
            required=False,
        ))

        self._gui = None
        self._wayland_controller = wayland_controller
        self._ipc_client = ipc_client

    def _get_ipc(self) -> HyprlandIPCClient:
        if self._ipc_client is None:
            self._ipc_client = HyprlandIPCClient()
        return self._ipc_client

    def _ensure_gui(self) -> None:
        """Initialize the unified Wayland/Linux input controller (or non-Linux pyautogui fallback)."""
        if self._wayland_controller is not None:
            return
        try:
            self._wayland_controller = WaylandInputController(auto_check=False)
            return
        except Exception as exc:
            logger.debug("WaylandInputController init failed, checking pyautogui fallback: %s", exc)

        if self._gui is None:
            try:
                import pyautogui
                pyautogui.FAILSAFE = True
                pyautogui.PAUSE = 0.25
                self._gui = pyautogui
            except (ImportError, OSError, RuntimeError) as exc:
                raise BackendUnavailableError(
                    f"Desktop input backend is unavailable: {exc}",
                ) from exc

    def _structured_error(
        self,
        *,
        error_type: str,
        error: str,
        remedy_hint: str = "",
        allowed_actions: Optional[List[str]] = None,
        **extra: Any,
    ) -> ToolResult:
        payload: Dict[str, Any] = {
            "status": "error",
            "error_type": error_type,
            "error": error,
            "remedy_hint": remedy_hint,
            "allowed_actions": list(allowed_actions or []),
        }
        payload.update(extra)
        return ToolResult(
            success=False,
            output=payload,
            error=error,
            error_type=error_type,
        )

    def _exception_to_tool_result(self, action: str, exc: BaseException) -> ToolResult:
        if isinstance(exc, InputControlError):
            allowed = {
                "DESKTOP_UNAVAILABLE": ["execute_command"],
                "BACKEND_UNAVAILABLE": ["execute_command"],
                "PERMISSION_DENIED": ["execute_command"],
                "INVALID_COORDINATES": ["capture_screen", "desktop_control", "manage_desktop_window"],
                "INVALID_INPUT": ["desktop_control"],
                "INPUT_TIMEOUT": ["manage_desktop_window", "capture_screen", "desktop_control"],
                "INPUT_FAILED": ["manage_desktop_window", "capture_screen", "desktop_control"],
            }.get(exc.error_type, ["manage_desktop_window", "desktop_control"])
            return self._structured_error(
                error_type=exc.error_type,
                error=str(exc),
                remedy_hint=exc.remedy_hint,
                allowed_actions=allowed,
                action=action,
            )
        return self._structured_error(
            error_type="INPUT_FAILED",
            error=f"Desktop automation error ({action}): {exc}",
            remedy_hint="Inspect active window and desktop environment before retrying.",
            allowed_actions=["manage_desktop_window", "capture_screen"],
            action=action,
        )

    async def _verify_expected_window_focus(self, expected_window: Optional[str]) -> Optional[ToolResult]:
        """Fail closed with FOCUS_LOST if expected_window does not match the active focused window."""
        exp = str(expected_window or "").strip()
        if not exp:
            return None
        ipc = self._get_ipc()
        if not ipc.can_connect() and not (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "").strip():
            return None
        try:
            active = await ipc.get_active_window()
        except Exception as exc:
            return self._structured_error(
                error_type="DESKTOP_UNAVAILABLE",
                error=f"Cannot verify focus for '{exp}': compositor IPC failed ({exc}).",
                remedy_hint="Check Hyprland session availability via manage_desktop_window.",
                allowed_actions=["execute_command"],
            )

        exp_l = exp.lower()
        win_addr = str(active.get("address") or "").lower()
        win_cls = str(active.get("class") or active.get("initialClass") or "").lower()
        win_title = str(active.get("title") or active.get("initialTitle") or "").lower()
        matched = (
            (exp_l.startswith("0x") and win_addr == exp_l)
            or (exp_l in win_title)
            or (exp_l in win_cls)
        )
        if not matched:
            return self._structured_error(
                error_type="FOCUS_LOST",
                error=(
                    f"FOCUS_LOST: Target window '{exp}' is not focused (currently active: "
                    f"title={active.get('title')!r}, class={active.get('class')!r}, address={active.get('address')!r})."
                ),
                remedy_hint=(
                    f"Call manage_desktop_window(action='focus', target='{exp}') to restore focus, "
                    "then retry desktop_control."
                ),
                allowed_actions=["manage_desktop_window", "capture_screen", "desktop_control"],
                expected_window=exp,
                active_window={
                    "address": active.get("address"),
                    "title": active.get("title"),
                    "class": active.get("class"),
                    "at": active.get("at"),
                    "size": active.get("size"),
                },
            )
        return None

    async def _get_monitor_list(self) -> List[Dict[str, Any]]:
        ipc = self._get_ipc()
        try:
            if hasattr(ipc, "get_monitors") and (ipc.can_connect() or (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "").strip()):
                return await ipc.get_monitors()
        except Exception:
            pass
        return []

    async def execute(self, action: Any = None, **kwargs: Any) -> ToolResult:
        """Execute a desktop observation or actuation command with policy and focus safety."""
        if isinstance(action, dict):
            merged = dict(action)
            merged.update(kwargs)
            eff_action = str(merged.pop("action", "") or "").strip().lower()
            params = merged
        else:
            eff_action = str(action or kwargs.pop("action", "") or "").strip().lower()
            params = dict(kwargs)

        from zelyr.engine.execution_policy import (
            authorized_policy_context,
            evaluate_direct_tool_policy_async,
        )

        pol_dec = await evaluate_direct_tool_policy_async(
            self.tool_id,
            {"action": eff_action, **params},
            tool_obj=self,
            schema=self.schema if hasattr(self, "schema") else None,
            required_ring=getattr(self, "required_ring", 0),
            is_registered=True,
            source_surface="desktop_control.execute",
        )
        if not pol_dec.allowed:
            err_p = pol_dec.to_tool_error_payload()
            return ToolResult(
                success=False,
                output=err_p,
                error=pol_dec.reason,
                error_type=err_p["error_type"],
                metadata={"policy_decision": pol_dec.to_safe_dict()},
            )

        # Preserve FastPath interception for high-level actions such as open_url
        try:
            from zelyr.core.fast_path_router import FastestPathRouter

            with authorized_policy_context(pol_dec):
                fast_path_result = FastestPathRouter.evaluate_and_intercept(eff_action, params)
            if fast_path_result and fast_path_result.get("intercepted"):
                cmd_str = " ".join(fast_path_result.get("command", []))
                logger.info("DesktopAutomationTool: Fast Path intercepted action '%s' -> %s", eff_action, cmd_str)
                return ToolResult(
                    success=fast_path_result.get("success", True),
                    output=fast_path_result.get("output", f"Executed CLI shortcut: {cmd_str}"),
                    metadata={"fast_path": True, "command": fast_path_result.get("command")},
                )
        except Exception as exc:
            logger.debug("Fast path router evaluation skipped: %s", exc)

        if eff_action not in _VALID_ACTIONS:
            return self._structured_error(
                error_type="INVALID_INPUT",
                error=f"Unknown action '{eff_action}'. Valid actions: {', '.join(sorted(_VALID_ACTIONS))}.",
                remedy_hint="Choose a supported desktop_control action such as get_screen_size, inspect_windows, observe, mouse_move, mouse_click, double_click, keyboard_type, or keyboard_press.",
                allowed_actions=["desktop_control"],
            )

        try:
            self._ensure_gui()
            if eff_action in _MUTATING_ACTIONS:
                expected_win = params.get("expected_window") or params.get("window")
                focus_err = await self._verify_expected_window_focus(expected_win)
                if focus_err is not None:
                    return focus_err

            handler = getattr(self, f"_action_{eff_action}")
            with authorized_policy_context(pol_dec):
                return await handler(**params)
        except asyncio.CancelledError:
            if self._wayland_controller is not None:
                self._wayland_controller.close()
            raise
        except Exception as exc:
            if self._wayland_controller is not None and self._wayland_controller._uinput is not None:
                self._wayland_controller._uinput.release_all()
            return self._exception_to_tool_result(eff_action, exc)

    # ── Read-Only Observation Actions ─────────────────────────────────── #

    async def _action_get_screen_size(self, **_: Any) -> ToolResult:
        ipc = self._get_ipc()
        if ipc.can_connect() or (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "").strip():
            try:
                geom = await ipc.get_desktop_geometry()
                monitors = geom.get("monitors") or []
                if monitors:
                    focused = geom.get("focused_monitor") or monitors[0]
                    cursor = await ipc.get_cursor_pos()
                    active = await ipc.get_active_window()
                    return ToolResult(
                        success=True,
                        output={
                            "status": "success",
                            "width": focused.get("logical_width") or focused.get("width"),
                            "height": focused.get("logical_height") or focused.get("height"),
                            "focused_monitor": focused,
                            "monitors": monitors,
                            "virtual_bounds": {
                                "min_x": geom["min_x"],
                                "min_y": geom["min_y"],
                                "max_x": geom["max_x"],
                                "max_y": geom["max_y"],
                                "width": geom["virtual_width"],
                                "height": geom["virtual_height"],
                            },
                            "cursor_pos": list(cursor) if cursor else None,
                            "active_window": {
                                "address": active.get("address"),
                                "title": active.get("title"),
                                "class": active.get("class"),
                                "at": active.get("at"),
                                "size": active.get("size"),
                            }
                            if active and active.get("address")
                            else None,
                            "message": (
                                f"Focused monitor {focused.get('name')}: {focused.get('logical_width')}x{focused.get('logical_height')} "
                                f"at ({focused.get('x')},{focused.get('y')}); virtual bounds [{geom['min_x']}..{geom['max_x']}, {geom['min_y']}..{geom['max_y']}]."
                            ),
                        },
                        metadata={"width": geom["virtual_width"], "height": geom["virtual_height"]},
                    )
            except Exception as exc:
                logger.debug("Hyprland geometry lookup failed: %s", exc)

        if self._gui is not None:
            loop = asyncio.get_running_loop()
            size = await loop.run_in_executor(None, self._gui.size)
            return ToolResult(
                success=True,
                output={"status": "success", "width": size.width, "height": size.height, "message": f"Screen size: {size.width}x{size.height} pixels"},
                metadata={"width": size.width, "height": size.height},
            )

        if not (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY") or os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")):
            raise DesktopUnavailableError("No active Wayland or X11 display session detected.")

        raise BackendUnavailableError("Could not query monitor geometry from compositor.")

    async def _action_inspect_windows(self, **_: Any) -> ToolResult:
        ipc = self._get_ipc()
        if not ipc.can_connect() and not (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "").strip():
            raise DesktopUnavailableError("Hyprland compositor session not detected.")
        clients = await ipc.get_clients()
        active = await ipc.get_active_window()
        monitors = await ipc.get_monitors()
        cursor = await ipc.get_cursor_pos()
        summaries = []
        for c in clients[:20]:
            at = c.get("at") if isinstance(c.get("at"), (list, tuple)) and len(c.get("at")) >= 2 else [0, 0]
            sz = c.get("size") if isinstance(c.get("size"), (list, tuple)) and len(c.get("size")) >= 2 else [0, 0]
            ws = c.get("workspace") if isinstance(c.get("workspace"), dict) else {}
            summaries.append({
                "address": c.get("address"),
                "title": c.get("title"),
                "class": c.get("class"),
                "at": [int(at[0]), int(at[1])],
                "size": [int(sz[0]), int(sz[1])],
                "center": [int(at[0]) + int(sz[0]) // 2, int(at[1]) + int(sz[1]) // 2],
                "workspace": ws.get("name") or ws.get("id"),
                "monitor": c.get("monitor"),
                "focused": bool(c.get("focused") or c.get("focusHistoryID") == 0),
            })
        return ToolResult(
            success=True,
            output={
                "status": "success",
                "active_window": active if isinstance(active, dict) and active.get("address") else None,
                "windows": summaries,
                "monitors": monitors,
                "cursor_pos": list(cursor) if cursor else None,
                "message": f"Inspected {len(summaries)} window(s) across {len(monitors)} monitor(s).",
            },
        )

    async def _action_get_active_window(self, **kwargs: Any) -> ToolResult:
        return await self._action_inspect_windows(**kwargs)

    async def _action_get_cursor_position(self, **_: Any) -> ToolResult:
        ipc = self._get_ipc()
        cpos = await ipc.get_cursor_pos()
        if cpos is None:
            raise BackendUnavailableError("Cursor position query requires an active Hyprland session.")
        return ToolResult(
            success=True,
            output={"status": "success", "cursor_pos": [cpos[0], cpos[1]], "x": cpos[0], "y": cpos[1]},
        )

    async def _action_observe(self, **kwargs: Any) -> ToolResult:
        target = str(kwargs.get("target") or ("window" if kwargs.get("window") else "active_monitor"))
        obs = await capture_desktop_observation(
            target=target,
            window=kwargs.get("window"),
            monitor=kwargs.get("monitor"),
            region=kwargs.get("region"),
            include_image=bool(kwargs.get("include_image", True)),
            ipc_client=self._get_ipc(),
        )
        v_b64 = obs.pop("_vision_b64", None)
        if not obs.get("success"):
            return self._structured_error(
                error_type=str(obs.get("error_type") or "SCREENSHOT_FAILED"),
                error=str(obs.get("error") or "Observation failed"),
                remedy_hint=str(obs.get("remedy_hint") or ""),
                allowed_actions=obs.get("allowed_actions") or ["manage_desktop_window"],
            )
        meta = {"_vision_b64": v_b64} if v_b64 else {}
        return ToolResult(success=True, output=obs, metadata=meta)

    async def _action_screenshot(self, **kwargs: Any) -> ToolResult:
        return await self._action_observe(**kwargs)

    # ── Actuation Actions ─────────────────────────────────────────────── #

    async def _action_mouse_move(self, x: Any = None, y: Any = None, **_: Any) -> ToolResult:
        if x is None or y is None:
            raise InvalidCoordinatesError("mouse_move requires both 'x' and 'y' parameters.")
        monitors = await self._get_monitor_list()
        loop = asyncio.get_running_loop()
        if self._wayland_controller is not None:
            ix, iy = await loop.run_in_executor(
                None,
                lambda: self._wayland_controller.mouse_move(x, y, monitors=monitors),
            )
        else:
            ix, iy = WaylandInputController(auto_check=False).validate_coordinates(x, y, monitors=monitors)
            await loop.run_in_executor(None, partial(self._gui.moveTo, ix, iy, duration=0.2))
        return ToolResult(
            success=True,
            output={"status": "success", "action": "mouse_move", "x": ix, "y": iy, "message": f"Mouse moved to ({ix}, {iy})."},
        )

    async def _action_mouse_click(
        self,
        button: str = "left",
        clicks: int = 1,
        x: Any = None,
        y: Any = None,
        **_: Any,
    ) -> ToolResult:
        if (x is None) != (y is None):
            raise InvalidCoordinatesError("Provide both 'x' and 'y' when clicking at explicit coordinates.")
        monitors = await self._get_monitor_list()
        loop = asyncio.get_running_loop()
        if self._wayland_controller is not None:
            await loop.run_in_executor(
                None,
                lambda: self._wayland_controller.mouse_click(button=button, clicks=clicks, x=x, y=y, monitors=monitors),
            )
            coords_msg = f" at ({int(round(float(x)))}, {int(round(float(y)))})" if x is not None and y is not None else ""
        else:
            if button not in ("left", "right", "middle"):
                raise InvalidInputKeyError("button must be 'left', 'right', or 'middle'.")
            await loop.run_in_executor(
                None,
                partial(
                    self._gui.click,
                    x=int(x) if x is not None else None,
                    y=int(y) if y is not None else None,
                    button=button,
                    clicks=int(clicks),
                ),
            )
            coords_msg = f" at ({x}, {y})" if x is not None else ""
        active_win = None
        cursor_pos = None
        try:
            ipc = self._get_ipc()
            if ipc.can_connect() or (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "").strip():
                act = await ipc.get_active_window()
                if isinstance(act, dict) and act.get("address"):
                    active_win = {
                        "title": act.get("title"),
                        "class": act.get("class"),
                        "at": act.get("at"),
                        "size": act.get("size"),
                    }
                cp = await ipc.get_cursor_pos()
                if cp is not None:
                    cursor_pos = [cp[0], cp[1]]
        except Exception:
            pass
        return ToolResult(
            success=True,
            output={
                "status": "success",
                "action": "mouse_click",
                "button": button,
                "clicks": clicks,
                "x": x,
                "y": y,
                "cursor_pos": cursor_pos,
                "active_window": active_win,
                "message": (
                    f"Clicked {button} button {clicks} time(s){coords_msg}. "
                    "Call capture_screen to observe updated UI state or confirm coordinates before clicking another control."
                ),
            },
        )

    async def _action_double_click(self, button: str = "left", x: Any = None, y: Any = None, **_: Any) -> ToolResult:
        return await self._action_mouse_click(button=button, clicks=2, x=x, y=y)

    async def _action_mouse_drag(
        self,
        x: Any = None,
        y: Any = None,
        end_x: Any = None,
        end_y: Any = None,
        button: str = "left",
        steps: int = 8,
        **_: Any,
    ) -> ToolResult:
        if x is None or y is None or end_x is None or end_y is None:
            raise InvalidCoordinatesError("mouse_drag requires 'x', 'y', 'end_x', and 'end_y'.")
        monitors = await self._get_monitor_list()
        loop = asyncio.get_running_loop()
        if self._wayland_controller is not None:
            await loop.run_in_executor(
                None,
                lambda: self._wayland_controller.mouse_drag(x, y, end_x, end_y, button=button, steps=steps, monitors=monitors),
            )
        else:
            sx, sy = WaylandInputController(auto_check=False).validate_coordinates(x, y, monitors=monitors)
            ex, ey = WaylandInputController(auto_check=False).validate_coordinates(end_x, end_y, monitors=monitors)
            await loop.run_in_executor(None, partial(self._gui.moveTo, sx, sy))
            await loop.run_in_executor(None, partial(self._gui.dragTo, ex, ey, button=button, duration=0.3))
        return ToolResult(
            success=True,
            output={
                "status": "success",
                "action": "mouse_drag",
                "from": [x, y],
                "to": [end_x, end_y],
                "message": f"Dragged {button} mouse from ({x}, {y}) to ({end_x}, {end_y}).",
            },
        )

    async def _action_mouse_scroll(
        self,
        direction: str = "down",
        amount: int = 3,
        delta_y: Optional[int] = None,
        delta_x: Optional[int] = None,
        x: Any = None,
        y: Any = None,
        **_: Any,
    ) -> ToolResult:
        dir_l = str(direction or "down").strip().lower()
        ticks = max(1, min(20, abs(int(amount or 3))))
        if delta_y is None and delta_x is None:
            if dir_l == "up":
                dy, dx = ticks, 0
            elif dir_l == "down":
                dy, dx = -ticks, 0
            elif dir_l == "left":
                dy, dx = 0, -ticks
            elif dir_l == "right":
                dy, dx = 0, ticks
            else:
                raise InvalidInputKeyError("direction must be 'up', 'down', 'left', or 'right'.")
        else:
            dy = int(delta_y or 0)
            dx = int(delta_x or 0)

        monitors = await self._get_monitor_list()
        loop = asyncio.get_running_loop()
        if self._wayland_controller is not None:
            await loop.run_in_executor(
                None,
                lambda: self._wayland_controller.mouse_scroll(delta_y=dy, delta_x=dx, x=x, y=y, monitors=monitors),
            )
        else:
            await loop.run_in_executor(None, partial(self._gui.scroll, dy))
        return ToolResult(
            success=True,
            output={"status": "success", "action": "mouse_scroll", "delta_y": dy, "delta_x": dx},
        )

    async def _action_keyboard_type(self, text: Any = None, press_enter: bool = False, **_: Any) -> ToolResult:
        if text is None or not isinstance(text, str) or not text:
            raise InvalidInputKeyError("keyboard_type requires a non-empty string 'text' parameter.")
        loop = asyncio.get_running_loop()
        if self._wayland_controller is not None:
            await loop.run_in_executor(None, self._wayland_controller.keyboard_type, text)
            if press_enter:
                await loop.run_in_executor(None, self._wayland_controller.keyboard_press, "enter")
        else:
            await loop.run_in_executor(None, partial(self._gui.write, text, interval=0.02))
            if press_enter:
                await loop.run_in_executor(None, partial(self._gui.press, "enter"))
        return ToolResult(
            success=True,
            output={
                "status": "success",
                "action": "keyboard_type",
                "length": len(text),
                "press_enter": bool(press_enter),
                "message": f"Typed {len(text)} character(s)" + (" and pressed Enter." if press_enter else "."),
            },
        )

    async def _action_keyboard_press(self, key: Any = None, **_: Any) -> ToolResult:
        if key is None or not isinstance(key, str) or not key.strip():
            raise InvalidInputKeyError("keyboard_press requires a non-empty 'key' parameter.")
        loop = asyncio.get_running_loop()
        if self._wayland_controller is not None:
            await loop.run_in_executor(None, self._wayland_controller.keyboard_press, key)
        else:
            key_map = {"super": "win", "windows": "win", "cmd": "win"}
            actual_key = key_map.get(key.lower(), key.lower())
            await loop.run_in_executor(None, partial(self._gui.press, actual_key))
        return ToolResult(
            success=True,
            output={"status": "success", "action": "keyboard_press", "key": key, "message": f"Pressed key '{key}'."},
        )

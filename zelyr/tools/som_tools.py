"""Zelyr Tools — Vimium-style SoM Click Tool.

Exposes two tools to the LLM orchestrator:

  capture_som_screen
      Captures the current desktop, runs the SoM overlay pipeline, and
      returns the annotated screenshot as a base64 string together with
      the list of visible tags.  The VLM should call this first to "see"
      the screen with tags before issuing any click_tag calls.

  click_tag
      Looks up a 2-letter tag (e.g. "AB") in the SoMManager's active_tags
      table and fires a pyautogui click at the stored (x, y) coordinate.
      Returns a plain-English success/error string to guide the next step.

Both tools share a single module-level SoMManager singleton so the tag table
produced by capture_som_screen remains valid until the next capture.
"""

import asyncio
import base64
import logging
import os
import subprocess
import tempfile
from functools import partial
from pathlib import Path
from typing import Optional

from zelyr.tools.core import BaseTool, ToolParameter, ToolResult
from zelyr.vision.som_manager import SoMManager
from zelyr.services.os_controller.factory import get_os_controller

logger = logging.getLogger(__name__)

# os_driver removed in favor of dynamic loading

# ── Singleton SoMManager ──────────────────────────────────────────────────────
# Shared between both tools so click_tag always sees the tags from the most
# recent capture_som_screen call, no matter which order the orchestrator
# invokes them.
_som: SoMManager = SoMManager()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — capture_som_screen
# ─────────────────────────────────────────────────────────────────────────────

class CaptureSomScreenTool(BaseTool):
    """Capture the desktop, overlay Vimium-style two-letter tags, return image.

    Call this tool before click_tag so the VLM can read the tags from the
    annotated screenshot and decide which element to interact with.

    Returns a JSON object with:
      - image_b64:  Base64-encoded PNG of the annotated screen.
      - tags:       List of all visible tag strings (e.g. ["AA", "AB", "AC"]).
      - tag_count:  Number of detected UI elements.
      - overlay_path: Filesystem path of the saved overlay image (optional use).
    """

    def __init__(self) -> None:
        super().__init__(
            tool_id="capture_som_screen",
            name="CaptureSomScreenTool",
            description=(
                "Captures the current desktop screen and overlays Vimium-style "
                "two-letter tags (AA, AB, AC …) on every detected UI element. "
                "Returns the annotated screenshot as a base64 image and a list "
                "of visible tags. Always call this tool FIRST before click_tag "
                "so you know which tags are available on the current screen."
            ),
        )

    async def execute(self, **_kwargs) -> ToolResult:  # type: ignore[override]
        """Capture screen, run SoM pipeline, return annotated image + tag list."""

        # ── Step 1: Capture raw screenshot ───────────────────────────────────
        raw_path = await self._capture_screen_to_file()
        if raw_path is None:
            return ToolResult(
                success=False,
                error=(
                    "Screen capture failed. Neither 'grim' (Wayland) nor "
                    "'scrot'/'import' (X11) is available, and python-mss "
                    "could not be imported.  Install one of those utilities."
                ),
            )

        # ── Step 2: Reset tags and run overlay ───────────────────────────────
        _som.clear_tags()                     # invalidate tags from last frame
        overlay_path = _som.generate_overlay(
            raw_path,
            output_path="/tmp/zelyr_som_overlay.png",
        )

        if overlay_path is None:
            # Overlay failed (missing cv2?) — still return raw screenshot so
            # the VLM is not left completely blind.
            logger.warning("SoM overlay failed; returning raw screenshot.")
            return await self._build_result_from_path(raw_path, overlay_ok=False)

        return await self._build_result_from_path(overlay_path, overlay_ok=True)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    async def _capture_screen_to_file() -> Optional[str]:
        """Capture the desktop to a temp PNG and return the path, or None."""
        out_path = os.path.join(tempfile.gettempdir(), "zelyr_som_raw.png")
        loop = asyncio.get_event_loop()

        # Wayland: try grim first.
        if os.environ.get("WAYLAND_DISPLAY"):
            if _which("grim"):
                logger.info("Triggering grim screenshot...")
                try:
                    ok = await loop.run_in_executor(
                        None,
                        partial(
                            subprocess.run,
                            ["grim", "-c", out_path],
                            capture_output=True,
                            timeout=3.0
                        ),
                )
                    if ok.returncode == 0:
                        return out_path
                except subprocess.TimeoutExpired:
                    logger.error("TimeoutExpired: grim screenshot hung")

        # X11 / generic: try scrot, then ImageMagick import.
        for cmd in (["scrot", out_path], ["import", "-window", "root", out_path]):
            if _which(cmd[0]):
                ok = await loop.run_in_executor(
                    None,
                    partial(subprocess.run, cmd, capture_output=True, timeout=3.0),
                )
                if ok.returncode == 0:
                    return out_path

        # Last resort: python-mss (works on Wayland with XWayland fallback).
        try:
            import mss  # type: ignore

            def _mss_grab():
                with mss.mss() as sct:
                    monitor = sct.monitors[1]   # primary monitor
                    sct_img = sct.grab(monitor)
                    mss.tools.to_png(sct_img.rgb, sct_img.size, output=out_path)

            await loop.run_in_executor(None, _mss_grab)
            return out_path
        except Exception as exc:
            logger.error("mss capture failed: %s", exc)

        return None

    @staticmethod
    async def _build_result_from_path(path: str, *, overlay_ok: bool) -> ToolResult:
        """Read the image at path, encode as base64, and build ToolResult."""
        loop = asyncio.get_event_loop()
        try:
            raw_bytes = await loop.run_in_executor(
                None, Path(path).read_bytes
            )
        except OSError as exc:
            return ToolResult(success=False, error=f"Could not read overlay file: {exc}")

        logger.info("Screenshot captured. Resizing image...")
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(raw_bytes))
            img = img.convert("RGB")
            
            max_size = 1024
            width, height = img.size
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int((height / width) * max_size)
                else:
                    new_height = max_size
                    new_width = int((width / height) * max_size)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
            logger.info("Image resized. Sending payload to Ollama VLM...")
            out_buffer = io.BytesIO()
            img.save(out_buffer, format="JPEG", quality=85)
            jpeg_bytes = out_buffer.getvalue()
            b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
            logger.info("VLM response received.")
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            b64 = base64.b64encode(raw_bytes).decode("utf-8")
        tags = sorted(_som.active_tags.keys())   # e.g. ["AA", "AB", "AC", …]

        return ToolResult(
            success=True,
            output={
                "image_b64": b64,
                "format": "png",
                "tags": tags,
                "tag_count": len(tags),
                "overlay_applied": overlay_ok,
                "overlay_path": path,
                "message": (
                    f"Screen captured with {len(tags)} tagged UI elements. "
                    "Read the tags from the image and call click_tag(tag=...) "
                    "to interact with any element."
                    if overlay_ok
                    else
                    "Raw screenshot returned (SoM overlay unavailable). "
                    "Install opencv-python for element tagging."
                ),
            },
            metadata={"tag_map": dict(_som.active_tags)},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — click_tag
# ─────────────────────────────────────────────────────────────────────────────

class SomClickTool(BaseTool):
    """Deterministically click a UI element by its two-letter SoM tag.

    The tag must come from the most recent capture_som_screen call.
    If the tag is not found, you must call capture_som_screen again to
    refresh the tag table before retrying.
    """

    def __init__(self) -> None:
        super().__init__(
            tool_id="click_tag",
            name="SomClickTool",
            description=(
                "Click a UI element identified by its two-letter Set-of-Mark "
                "tag (e.g. 'AB'). The tag must appear in the most recent "
                "capture_som_screen output. If the tag is not found, call "
                "capture_som_screen again to get an updated tag map."
            ),
        )
        self.add_parameter(ToolParameter(
            name="tag",
            type="string",
            description=(
                "The two-letter tag printed on the UI element you want to click "
                "(e.g. 'AA', 'AB', 'BC'). Case-insensitive."
            ),
            required=True,
        ))
        self.add_parameter(ToolParameter(
            name="button",
            type="string",
            description="Which mouse button to use: 'left' (default) or 'right'.",
            required=False,
            default="left",
        ))
        self.add_parameter(ToolParameter(
            name="clicks",
            type="integer",
            description="Number of clicks (1 = single-click, 2 = double-click). Default: 1.",
            required=False,
            default=1,
        ))

    # ── Public execute ────────────────────────────────────────────────────────

    async def execute(  # type: ignore[override]
        self,
        tag: str,
        button: str = "left",
        clicks: int = 1,
        **_kwargs,
    ) -> ToolResult:
        """Look up tag in active_tags and click the stored coordinate."""

        # ── Validate tag format ───────────────────────────────────────────────
        tag = tag.strip().upper()
        if len(tag) != 2 or not tag.isalpha():
            return ToolResult(
                success=False,
                error=(
                    f"Invalid tag '{tag}'. Tags are exactly two uppercase letters "
                    "(e.g. 'AA', 'BC'). Call capture_som_screen to see current tags."
                ),
            )

        # ── Validate button ───────────────────────────────────────────────────
        if button not in ("left", "right"):
            return ToolResult(
                success=False,
                error=f"Invalid button '{button}'. Use 'left' or 'right'.",
            )

        # ── Look up coordinate ────────────────────────────────────────────────
        coord = _som.lookup(tag)
        if coord is None:
            # Give the VLM a clear, actionable error message.
            available = sorted(_som.active_tags.keys())
            return ToolResult(
                success=False,
                error=(
                    f"Tag '{tag}' not found in the current tag map. "
                    f"Available tags: {available if available else 'none — call capture_som_screen first'}. "
                    "The screen may have changed; call capture_som_screen again."
                ),
                metadata={"available_tags": available},
            )

        x, y = coord

        controller = get_os_controller()
        if not controller.can_click:
            return ToolResult(
                success=False,
                error="Error: Mouse automation is not supported on this specific Wayland compositor. Do not attempt to click again.",
            )

        # ── Perform the click ─────────────────────────────────────────────────
        try:
            controller.click(x, y, button, clicks)
        except Exception as exc:
            logger.error("SomClickTool: click failed at (%d, %d): %s", x, y, exc)
            return ToolResult(
                success=False,
                error=f"Click at ({x}, {y}) failed: {exc}",
            )

        logger.info(
            "SomClickTool: %s-clicked tag %s at (%d, %d) × %d",
            button, tag, x, y, clicks,
        )
        return ToolResult(
            success=True,
            output=(
                f"Successfully {button}-clicked element '{tag}' at ({x}, {y})"
                + (f" × {clicks}" if clicks > 1 else "") + "."
            ),
            metadata={"tag": tag, "x": x, "y": y, "button": button, "clicks": clicks},
        )

    # ── Private helpers ───────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — type_text
# ─────────────────────────────────────────────────────────────────────────────

class SomTypeTool(BaseTool):
    """Type text deterministically via Wayland's wtype."""

    def __init__(self) -> None:
        super().__init__(
            tool_id="type_text",
            name="SomTypeTool",
            description="Type a string of text into the currently focused window using Wayland's wtype.",
        )
        self.add_parameter(ToolParameter(
            name="text",
            type="string",
            description="The text to type.",
            required=True,
        ))

    async def execute(self, text: str, **_kwargs) -> ToolResult:  # type: ignore[override]
        controller = get_os_controller()
        if not controller.can_type:
            return ToolResult(
                success=False,
                error="Error: Keyboard automation is not supported on this specific Wayland compositor. Do not attempt to type again."
            )
            
        try:
            controller.type_text(text)
            return ToolResult(
                success=True,
                output=f"Successfully typed: '{text}'"
            )
        except Exception as exc:
            logger.error("SomTypeTool: failed to type: %s", exc)
            return ToolResult(
                success=False,
                error=f"Failed to type text: {exc}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — press_key
# ─────────────────────────────────────────────────────────────────────────────

class SomKeyTool(BaseTool):
    """Press a specific key via Wayland's wtype."""

    def __init__(self) -> None:
        super().__init__(
            tool_id="press_key",
            name="SomKeyTool",
            description="Press a specific keyboard key (e.g., 'Return', 'Tab', 'Escape') into the currently focused window using Wayland's wtype.",
        )
        self.add_parameter(ToolParameter(
            name="key",
            type="string",
            description="The key to press (e.g., 'Return', 'Tab').",
            required=True,
        ))

    async def execute(self, key: str, **_kwargs) -> ToolResult:  # type: ignore[override]
        controller = get_os_controller()
        if not controller.can_type:
            return ToolResult(
                success=False,
                error="Error: Keyboard automation is not supported on this specific Wayland compositor. Do not attempt to press keys again."
            )
            
        try:
            controller.press_key(key)
            return ToolResult(
                success=True,
                output=f"Successfully pressed key: '{key}'"
            )
        except Exception as exc:
            logger.error("SomKeyTool: failed to press key: %s", exc)
            return ToolResult(
                success=False,
                error=f"Failed to press key: {exc}"
            )

# ── Utility ───────────────────────────────────────────────────────────────────

def _which(cmd: str) -> bool:
    """Return True if `cmd` exists on PATH."""
    import shutil
    return shutil.which(cmd) is not None


# ── Public singleton accessors (for external code that needs the shared state) ─

def get_som_manager() -> SoMManager:
    """Return the module-level SoMManager singleton."""
    return _som

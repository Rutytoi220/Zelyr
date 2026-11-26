"""Wayland-native desktop observation and structured screenshot capture."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from zelyr.engine.hyprland_ipc import HyprlandIPCClient
from zelyr.sdk.plugin import tool
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - handled cleanly at runtime
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_LAST_FRAME_LOCK = threading.Lock()
_LAST_FRAME_STATE: Dict[str, Tuple[str, bytes]] = {}
_MAX_CACHED_FRAMES = 16
_MAX_SAVED_SCREENSHOTS = 10
_MAX_VISION_B64_BYTES = 220_000
_SCREENSHOT_DIR = Path(tempfile.gettempdir()) / "zelyr_screenshots"


def _error_payload(
    error_type: str,
    error: str,
    remedy_hint: str,
    allowed_actions: Optional[List[str]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a canonical structured error dictionary for desktop observation failures."""
    payload: Dict[str, Any] = {
        "success": False,
        "status": "error",
        "error_type": error_type,
        "error": error,
        "remedy_hint": remedy_hint,
        "allowed_actions": list(allowed_actions or []),
    }
    payload.update(extra)
    return payload


def _has_desktop_display() -> bool:
    """Return True if a Wayland or X11 display environment variable is active."""
    return bool(
        (os.environ.get("WAYLAND_DISPLAY") or "").strip()
        or (os.environ.get("DISPLAY") or "").strip()
        or (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "").strip()
    )


def _prune_old_screenshots(directory: Path, keep: int = _MAX_SAVED_SCREENSHOTS) -> None:
    """Retain at most `keep` most recent screenshot files in the bounded scratch directory."""
    try:
        files = sorted(
            (p for p in directory.iterdir() if p.is_file() and p.name.startswith("zelyr_screen_")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in files[keep:]:
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass


def _parse_region(region: str) -> Optional[Tuple[int, int, int, int]]:
    """Parse 'x,y wxh' region geometry strictly."""
    if not isinstance(region, str):
        return None
    m = re.fullmatch(
        r"\s*(-?\d+)\s*,\s*(-?\d+)\s+(\d+)\s*x\s*(\d+)\s*",
        region,
    )
    if not m:
        return None
    x, y, w, h = (int(m.group(i)) for i in range(1, 5))
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _normalize_monitor_record(mon: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a Hyprland monitor dictionary with logical dimensions."""
    scale = float(mon.get("scale") or 1.0)
    if scale <= 0:
        scale = 1.0
    px_w = int(mon.get("width") or 0)
    px_h = int(mon.get("height") or 0)
    log_w = max(1, int(round(px_w / scale))) if px_w > 0 else 0
    log_h = max(1, int(round(px_h / scale))) if px_h > 0 else 0
    ws = mon.get("activeWorkspace") if isinstance(mon.get("activeWorkspace"), dict) else {}
    return {
        "id": mon.get("id"),
        "name": str(mon.get("name") or ""),
        "x": int(mon.get("x") or 0),
        "y": int(mon.get("y") or 0),
        "width": px_w,
        "height": px_h,
        "logical_width": log_w,
        "logical_height": log_h,
        "scale": scale,
        "focused": bool(mon.get("focused")),
        "active_workspace": ws.get("name") or ws.get("id"),
    }


def _summarize_window(win: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact, geometry-accurate summary of a compositor window."""
    at = win.get("at") if isinstance(win.get("at"), (list, tuple)) and len(win.get("at")) >= 2 else [0, 0]
    size = win.get("size") if isinstance(win.get("size"), (list, tuple)) and len(win.get("size")) >= 2 else [0, 0]
    wx, wy = int(at[0]), int(at[1])
    ww, wh = int(size[0]), int(size[1])
    ws = win.get("workspace") if isinstance(win.get("workspace"), dict) else {}
    return {
        "address": str(win.get("address") or ""),
        "title": str(win.get("title") or "")[:120],
        "class": str(win.get("class") or win.get("initialClass") or ""),
        "at": [wx, wy],
        "size": [ww, wh],
        "center": [wx + ww // 2, wy + wh // 2],
        "workspace": ws.get("name") or ws.get("id") or win.get("workspace"),
        "monitor": win.get("monitor"),
        "focused": bool(win.get("focused") or win.get("focusHistoryID") == 0),
        "floating": bool(win.get("floating")),
    }


def _run_capture_subprocess(cmd: List[str], timeout: float) -> Tuple[int, bytes, str]:
    """Execute a screenshot CLI command with bounded timeout and process-group cleanup."""
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout_b, stderr_b = proc.communicate(timeout=timeout)
        return proc.returncode, stdout_b, (stderr_b or b"").decode("utf-8", errors="replace").strip()
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.communicate(timeout=1.0)
        except Exception:
            pass
        raise


def _overlay_coordinate_grid(img: Any, region: Dict[str, Any]) -> Any:
    """Render subtle labeled desktop-coordinate gridlines on a preview copy for VLM accuracy."""
    if Image is None or ImageDraw is None or not region:
        return img
    try:
        rx = int(region.get("x", 0))
        ry = int(region.get("y", 0))
        rw = int(region.get("width") or img.width)
        rh = int(region.get("height") or img.height)
        if rw <= 60 or rh <= 60 or img.width < 120 or img.height < 120:
            return img
        step = 100 if max(rw, rh) <= 950 else 200
        minor_step = max(50, step // 2)
        overlay = img.convert("RGBA")
        grid = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(grid)

        font = None
        if ImageFont is not None:
            try:
                font = ImageFont.load_default(size=14 if max(img.width, img.height) <= 1000 else 16)
            except TypeError:
                font = ImageFont.load_default()

        def _measure(text: str) -> Tuple[int, int]:
            try:
                box = draw.textbbox((0, 0), text, font=font)
                return max(12, box[2] - box[0]), max(10, box[3] - box[1])
            except Exception:
                return len(text) * 7, 12

        # Minor faint subgrid lines (50px on windows, 100px on full monitors)
        minor_x0 = ((rx + minor_step - 1) // minor_step) * minor_step
        show_minor_x_lbl = rw <= 700
        for dx in range(minor_x0, rx + rw, minor_step):
            if dx % step == 0:
                continue
            px = int(round((dx - rx) * img.width / rw))
            if 0 < px < img.width - 1:
                draw.line([(px, 0), (px, img.height - 1)], fill=(56, 189, 248, 65), width=1)
                if show_minor_x_lbl:
                    lbl = str(dx)
                    tw, th = _measure(lbl)
                    px_per_minor = (minor_step * img.width) / rw
                    if px_per_minor >= tw + 8:
                        bx = min(max(2, px + 3), max(2, img.width - tw - 6))
                        draw.rectangle([bx - 2, 2, bx + tw + 3, th + 5], fill=(13, 17, 23, 185))
                        draw.text((bx, 3), lbl, font=font, fill=(148, 224, 255, 220))

        minor_y0 = ((ry + minor_step - 1) // minor_step) * minor_step
        show_minor_y_lbl = rh <= 700
        for dy in range(minor_y0, ry + rh, minor_step):
            if dy % step == 0:
                continue
            py = int(round((dy - ry) * img.height / rh))
            if 0 < py < img.height - 1:
                draw.line([(0, py), (img.width - 1, py)], fill=(56, 189, 248, 65), width=1)
                if show_minor_y_lbl:
                    lbl = str(dy)
                    tw, th = _measure(lbl)
                    by = min(max(22, py + 2), max(22, img.height - th - 4))
                    draw.rectangle([2, by - 1, tw + 7, by + th + 2], fill=(13, 17, 23, 175))
                    draw.text((4, by), lbl, font=font, fill=(148, 224, 255, 220))

        # Major grid lines + bold readable axis coordinate labels
        first_x = ((rx + step - 1) // step) * step
        for dx in range(first_x, rx + rw, step):
            px = int(round((dx - rx) * img.width / rw))
            if 0 < px < img.width - 1:
                draw.line([(px, 0), (px, img.height - 1)], fill=(56, 189, 248, 120), width=1)
                lbl = str(dx)
                tw, th = _measure(lbl)
                bx = min(max(2, px + 3), max(2, img.width - tw - 6))
                draw.rectangle([bx - 2, 2, bx + tw + 3, th + 5], fill=(13, 17, 23, 215))
                draw.text((bx, 3), lbl, font=font, fill=(125, 211, 252, 255))

        first_y = ((ry + step - 1) // step) * step
        for dy in range(first_y, ry + rh, step):
            py = int(round((dy - ry) * img.height / rh))
            if 0 < py < img.height - 1:
                draw.line([(0, py), (img.width - 1, py)], fill=(56, 189, 248, 120), width=1)
                lbl = str(dy)
                tw, th = _measure(lbl)
                by = min(max(22, py + 2), max(22, img.height - th - 4))
                draw.rectangle([2, by - 1, tw + 7, by + th + 2], fill=(13, 17, 23, 215))
                draw.text((4, by), lbl, font=font, fill=(125, 211, 252, 255))

        return Image.alpha_composite(overlay, grid).convert("RGB")
    except Exception:
        return img


def _update_frame_diff(cache_key: str, sha256: str, thumb_bytes: bytes) -> Tuple[Optional[bool], Optional[float]]:
    """Compare the current screenshot hash/thumbnail against the last capture for this target."""
    with _LAST_FRAME_LOCK:
        prev = _LAST_FRAME_STATE.get(cache_key)
        if len(_LAST_FRAME_STATE) >= _MAX_CACHED_FRAMES and cache_key not in _LAST_FRAME_STATE:
            oldest = next(iter(_LAST_FRAME_STATE))
            _LAST_FRAME_STATE.pop(oldest, None)
        _LAST_FRAME_STATE[cache_key] = (sha256, thumb_bytes)

    if prev is None:
        return None, None
    prev_sha, prev_thumb = prev
    if not prev_thumb or len(prev_thumb) != len(thumb_bytes):
        return sha256 != prev_sha, 1.0 if sha256 != prev_sha else 0.0
    total_delta = sum(abs(a - b) for a, b in zip(prev_thumb, thumb_bytes))
    diff_ratio = round(total_delta / (255.0 * len(thumb_bytes)), 5)
    changed = (sha256 != prev_sha) and (diff_ratio > 0.0002)
    return changed, diff_ratio


async def capture_desktop_observation(
    *,
    target: str = "active_monitor",
    window: Optional[str] = None,
    monitor: Optional[str] = None,
    region: Optional[str] = None,
    include_image: bool = True,
    max_edge: int = 960,
    timeout: float = 4.0,
    save_artifact: bool = True,
    runner: Optional[Callable[[List[str], float], Tuple[int, bytes, str]]] = None,
    ipc_client: Optional[HyprlandIPCClient] = None,
) -> Dict[str, Any]:
    """Capture a bounded desktop screenshot and compositor metadata observation."""
    t0 = time.monotonic()

    if Image is None:
        return _error_payload(
            "BACKEND_UNAVAILABLE",
            "Pillow (PIL) is not installed; cannot process desktop screenshots.",
            "Ensure Pillow is installed in the Zelyr Python environment.",
            ["execute_command"],
        )

    if runner is None and not _has_desktop_display():
        return _error_payload(
            "DESKTOP_UNAVAILABLE",
            "No active Wayland or X11 desktop session detected (WAYLAND_DISPLAY and DISPLAY are unset).",
            "Run inside an active desktop session or verify WAYLAND_DISPLAY / XDG_RUNTIME_DIR.",
            ["execute_command"],
        )

    backend: Optional[str] = None
    if shutil.which("grim") and ((os.environ.get("WAYLAND_DISPLAY") or "").strip() or (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "").strip()):
        backend = "grim"
    elif shutil.which("grim"):
        backend = "grim"
    elif shutil.which("spectacle"):
        backend = "spectacle"
    elif shutil.which("scrot"):
        backend = "scrot"

    if backend is None and runner is None:
        return _error_payload(
            "BACKEND_UNAVAILABLE",
            "No supported screenshot utility found ('grim', 'spectacle', or 'scrot').",
            "Install or expose 'grim' on Wayland or 'scrot' on X11.",
            ["execute_command"],
        )
    backend = backend or "grim"

    client = ipc_client or HyprlandIPCClient()
    monitors: List[Dict[str, Any]] = []
    clients: List[Dict[str, Any]] = []
    active_win_raw: Optional[Dict[str, Any]] = None
    cursor_pos: Optional[List[int]] = None

    if client.can_connect() or shutil.which("hyprctl"):
        try:
            raw_monitors = await client.get_monitors() if hasattr(client, "get_monitors") else await client._execute_json_command("j/monitors")
            if isinstance(raw_monitors, list):
                monitors = [_normalize_monitor_record(m) for m in raw_monitors if isinstance(m, dict)]
        except Exception as exc:
            logger.debug("Monitor query failed: %s", exc)
        try:
            clients = await client.get_clients()
        except Exception as exc:
            logger.debug("Client list query failed: %s", exc)
        try:
            active_win_raw = await client.get_active_window()
        except Exception as exc:
            logger.debug("Active window query failed: %s", exc)
        try:
            if hasattr(client, "get_cursor_pos"):
                cpos = await client.get_cursor_pos()
                if cpos is not None:
                    cursor_pos = [cpos[0], cpos[1]]
        except Exception:
            pass

    eff_target = (target or "active_monitor").strip().lower()
    if window and eff_target in ("active_monitor", "full"):
        eff_target = "window"
    if monitor and eff_target == "active_monitor":
        eff_target = "monitor"
    if region and eff_target in ("active_monitor", "full"):
        eff_target = "region"

    desktop_bounds: Optional[Dict[str, int]] = None
    if monitors:
        min_x = min(m["x"] for m in monitors)
        min_y = min(m["y"] for m in monitors)
        max_x = max(m["x"] + m["logical_width"] for m in monitors)
        max_y = max(m["y"] + m["logical_height"] for m in monitors)
        desktop_bounds = {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "virtual_width": max_x - min_x,
            "virtual_height": max_y - min_y,
        }

    captured_region: Dict[str, Any] = {}
    grim_args: List[str] = []

    if eff_target == "region":
        parsed = _parse_region(region or "")
        if parsed is None:
            return _error_payload(
                "INVALID_COORDINATES",
                f"Invalid region format '{region}'. Expected 'x,y wxh' with positive width and height.",
                "Pass region as 'x,y widthxheight' (for example '0,1080 1280x720').",
                ["desktop_control", "manage_desktop_window"],
            )
        rx, ry, rw, rh = parsed
        grim_args = ["-g", f"{rx},{ry} {rw}x{rh}"]
        captured_region = {"x": rx, "y": ry, "width": rw, "height": rh, "scale": 1.0}
    elif eff_target in ("window", "active_window"):
        matched_win: Optional[Dict[str, Any]] = None
        if eff_target == "window" and window:
            matched_win = HyprlandIPCClient._match_client(clients, window)
            if matched_win is None and active_win_raw:
                active_title = str(active_win_raw.get("title") or "").lower()
                active_cls = str(active_win_raw.get("class") or "").lower()
                if window.lower() in active_title or window.lower() in active_cls:
                    matched_win = active_win_raw
            if matched_win is None:
                return _error_payload(
                    "WINDOW_NOT_FOUND",
                    f"Target window '{window}' was not found among open compositor windows.",
                    "Call manage_desktop_window(action='list') to inspect open window titles and classes.",
                    ["manage_desktop_window"],
                )
        else:
            matched_win = active_win_raw

        if not matched_win or not isinstance(matched_win.get("at"), (list, tuple)) or not isinstance(matched_win.get("size"), (list, tuple)):
            return _error_payload(
                "WINDOW_NOT_FOUND",
                "No active window with valid geometry was found to capture.",
                "Focus a window or capture target='active_monitor' instead.",
                ["manage_desktop_window", "capture_screen"],
            )
        wx, wy = int(matched_win["at"][0]), int(matched_win["at"][1])
        ww, wh = int(matched_win["size"][0]), int(matched_win["size"][1])
        if ww <= 0 or wh <= 0:
            return _error_payload(
                "SCREENSHOT_FAILED",
                f"Target window has degenerate geometry ({ww}x{wh}).",
                "Ensure the window is mapped and visible before capturing.",
                ["manage_desktop_window"],
            )
        grim_args = ["-g", f"{wx},{wy} {ww}x{wh}"]
        win_summary = _summarize_window(matched_win)
        captured_region = {
            "x": wx,
            "y": wy,
            "width": ww,
            "height": wh,
            "scale": 1.0,
            "window": win_summary["title"],
        }
    elif eff_target in ("monitor", "active_monitor"):
        chosen_mon: Optional[Dict[str, Any]] = None
        if monitor:
            chosen_mon = next((m for m in monitors if m["name"].lower() == monitor.strip().lower()), None)
        if chosen_mon is None and active_win_raw is not None:
            win_mon_id = active_win_raw.get("monitor")
            if win_mon_id is not None:
                chosen_mon = next((m for m in monitors if m.get("id") == win_mon_id), None)
        if chosen_mon is None:
            chosen_mon = next((m for m in monitors if m.get("focused")), monitors[0] if monitors else None)
        if chosen_mon and chosen_mon.get("name"):
            grim_args = ["-o", str(chosen_mon["name"])]
            captured_region = {
                "x": int(chosen_mon["x"]),
                "y": int(chosen_mon["y"]),
                "width": int(chosen_mon["logical_width"]),
                "height": int(chosen_mon["logical_height"]),
                "monitor": chosen_mon["name"],
                "scale": chosen_mon["scale"],
            }
        elif desktop_bounds:
            captured_region = {
                "x": desktop_bounds["min_x"],
                "y": desktop_bounds["min_y"],
                "width": desktop_bounds["virtual_width"],
                "height": desktop_bounds["virtual_height"],
                "scale": 1.0,
            }
    elif eff_target == "full":
        if desktop_bounds:
            captured_region = {
                "x": desktop_bounds["min_x"],
                "y": desktop_bounds["min_y"],
                "width": desktop_bounds["virtual_width"],
                "height": desktop_bounds["virtual_height"],
                "scale": 1.0,
            }
    else:
        return _error_payload(
            "INVALID_INPUT",
            f"Unsupported capture target '{target}'. Expected 'active_monitor', 'active_window', 'window', 'monitor', 'region', or 'full'.",
            "Use target='active_monitor', target='active_window', or target='window'.",
            ["capture_screen"],
        )

    fd, raw_tmp_path = tempfile.mkstemp(prefix="zelyr_raw_cap_", suffix=".png")
    os.close(fd)
    exec_fn = runner or _run_capture_subprocess
    try:
        if backend == "grim":
            cmd = ["grim", *grim_args, raw_tmp_path]
        elif backend == "spectacle":
            cmd = ["spectacle", "-b", "-n", "-o", raw_tmp_path]
        else:
            cmd = ["scrot", raw_tmp_path]

        try:
            rc, _stdout_b, stderr_txt = await asyncio.to_thread(exec_fn, cmd, float(timeout))
        except subprocess.TimeoutExpired:
            return _error_payload(
                "SCREENSHOT_FAILED",
                f"Screenshot command '{cmd[0]}' timed out after {timeout}s.",
                "Verify desktop compositor responsiveness or target a single monitor/window.",
                ["manage_desktop_window", "execute_command"],
            )
        except OSError as exc:
            return _error_payload(
                "SCREENSHOT_FAILED",
                f"Failed to invoke screenshot backend '{cmd[0]}': {exc}",
                "Inspect desktop environment and display permissions.",
                ["execute_command"],
            )

        if rc != 0:
            return _error_payload(
                "SCREENSHOT_FAILED",
                f"Screenshot command '{cmd[0]}' exited with status {rc}: {stderr_txt or 'unknown error'}",
                "Verify target window/monitor exists and display session is unlocked.",
                ["manage_desktop_window", "execute_command"],
            )

        if not os.path.exists(raw_tmp_path) or os.path.getsize(raw_tmp_path) == 0:
            return _error_payload(
                "SCREENSHOT_FAILED",
                "Screenshot backend completed without producing image data.",
                "Retry capture or verify target monitor/window geometry.",
                ["manage_desktop_window", "capture_screen"],
            )

        try:
            with Image.open(raw_tmp_path) as opened_img:
                opened_img.load()
                orig_w, orig_h = opened_img.size
                rgb_img = opened_img.convert("RGB")
        except Exception as exc:
            return _error_payload(
                "SCREENSHOT_FAILED",
                f"Captured screenshot could not be decoded cleanly: {exc}",
                "Retry capture_screen or inspect compositor state.",
                ["capture_screen"],
            )

        eff_max_edge = max(128, min(1600, int(max_edge or 960)))
        if max(orig_w, orig_h) > eff_max_edge:
            if orig_w >= orig_h:
                new_w = eff_max_edge
                new_h = max(1, int(round((orig_h / orig_w) * eff_max_edge)))
            else:
                new_h = eff_max_edge
                new_w = max(1, int(round((orig_w / orig_h) * eff_max_edge)))
            resized = rgb_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            new_w, new_h = orig_w, orig_h
            resized = rgb_img

        thumb_bytes = resized.resize((64, 64), Image.Resampling.BILINEAR).tobytes()
        sha256_hex = hashlib.sha256(resized.tobytes()).hexdigest()
        cache_key = f"{eff_target}:{window or ''}:{monitor or ''}:{region or ''}"
        changed_since_last, diff_ratio = _update_frame_diff(cache_key, sha256_hex, thumb_bytes)

        if not captured_region:
            captured_region = {"x": 0, "y": 0, "width": orig_w, "height": orig_h, "scale": 1.0}

        annotated_img = _overlay_coordinate_grid(resized, captured_region) if include_image else resized

        buf = io.BytesIO()
        quality = 80
        annotated_img.save(buf, format="JPEG", quality=quality)
        jpeg_bytes = buf.getvalue()
        while len(jpeg_bytes) > _MAX_VISION_B64_BYTES and quality > 45:
            quality -= 12
            buf = io.BytesIO()
            annotated_img.save(buf, format="JPEG", quality=quality)
            jpeg_bytes = buf.getvalue()

        artifact_path: Optional[str] = None
        if save_artifact:
            try:
                _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
                _prune_old_screenshots(_SCREENSHOT_DIR)
                art_file = _SCREENSHOT_DIR / f"zelyr_screen_{int(time.time() * 1000)}_{os.getpid()}.jpg"
                art_file.write_bytes(jpeg_bytes)
                artifact_path = str(art_file)
            except OSError:
                artifact_path = None

        capture_ms = round((time.monotonic() - t0) * 1000.0, 2)
        active_summary = _summarize_window(active_win_raw) if isinstance(active_win_raw, dict) and active_win_raw.get("address") else None
        rx = int(captured_region.get("x", 0))
        ry = int(captured_region.get("y", 0))
        rw = int(captured_region.get("width") or orig_w)
        rh = int(captured_region.get("height") or orig_h)
        cx, cy = rx + rw // 2, ry + rh // 2

        if eff_target in ("window", "active_window", "region"):
            compact_active = (
                {k: active_summary[k] for k in ("title", "class") if k in active_summary}
                if isinstance(active_summary, dict)
                else None
            )
            obs: Dict[str, Any] = {
                "success": True,
                "status": "success",
                "backend": backend,
                "target": eff_target,
                "captured_region": captured_region,
                "image_size": [new_w, new_h],
                "cursor_pos": cursor_pos,
                "active_window": compact_active,
                "changed_since_last": changed_since_last,
                "diff_ratio": diff_ratio,
                "capture_ms": capture_ms,
                "image_path": artifact_path,
                "message": (
                    f"Captured {eff_target} ({orig_w}x{orig_h}) at desktop rect "
                    f"x={rx}..{rx + rw}, y={ry}..{ry + rh} (center=({cx}, {cy})). "
                    "Read cyan coordinate labels in global desktop (x, y) and pass them to desktop_control."
                ),
            }
        else:
            scoped_mons = monitors
            scoped_wins = [_summarize_window(c) for c in clients[:6] if isinstance(c, dict)]
            obs = {
                "success": True,
                "status": "success",
                "backend": backend,
                "target": eff_target,
                "captured_region": captured_region,
                "pixel_size": [orig_w, orig_h],
                "image_size": [new_w, new_h],
                "desktop_bounds": desktop_bounds,
                "monitors": scoped_mons,
                "cursor_pos": cursor_pos,
                "active_window": active_summary,
                "visible_windows": scoped_wins,
                "sha256": sha256_hex,
                "changed_since_last": changed_since_last,
                "diff_ratio": diff_ratio,
                "capture_ms": capture_ms,
                "image_bytes": len(jpeg_bytes),
                "image_path": artifact_path,
                "message": (
                    f"Captured {eff_target} screenshot ({orig_w}x{orig_h} -> {new_w}x{new_h}, {capture_ms}ms) "
                    f"covering desktop x={rx}..{rx + rw}, y={ry}..{ry + rh}. "
                    "Read cyan grid coordinates in global desktop (x, y)."
                ),
            }
        if include_image:
            b64_data = base64.b64encode(jpeg_bytes).decode("ascii")
            if len(b64_data) <= _MAX_VISION_B64_BYTES * 2:
                obs["_vision_b64"] = b64_data
        return obs
    finally:
        try:
            if os.path.exists(raw_tmp_path):
                os.unlink(raw_tmp_path)
        except OSError:
            pass


def capture_desktop_observation_sync(**kwargs: Any) -> Dict[str, Any]:
    """Synchronous wrapper for capture_desktop_observation."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(capture_desktop_observation(**kwargs))
    result_box: List[Dict[str, Any]] = []
    err_box: List[BaseException] = []

    def _worker() -> None:
        try:
            result_box.append(asyncio.run(capture_desktop_observation(**kwargs)))
        except BaseException as exc:  # pragma: no cover
            err_box.append(exc)

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    th.join(timeout=float(kwargs.get("timeout") or 4.0) + 2.0)
    if err_box:
        raise err_box[0]
    if not result_box:
        return _error_payload("SCREENSHOT_FAILED", "Screenshot capture thread timed out.", "Retry capture_screen.")
    return result_box[0]


@tool(
    name="capture_screen",
    description=(
        "Capture a screenshot and compositor window/monitor geometry observation "
        "(read-only). Supports target='active_monitor', 'active_window', 'window', 'monitor', 'region', or 'full'."
    ),
)
def capture_screen(
    target: str = "active_monitor",
    window: Optional[str] = None,
    monitor: Optional[str] = None,
    region: Optional[str] = None,
    include_image: bool = True,
    max_edge: int = 960,
) -> str:
    """Capture a desktop screenshot + geometry metadata and return serialized JSON."""
    obs = capture_desktop_observation_sync(
        target=target,
        window=window,
        monitor=monitor,
        region=region,
        include_image=include_image,
        max_edge=max_edge,
    )
    return json.dumps(obs)


class CaptureScreenTool(BaseTool):
    """Structured read-only desktop screen and compositor state observation tool."""

    read_only = True
    required_ring = 1

    def __init__(self, ipc_client: Optional[HyprlandIPCClient] = None):
        super().__init__(
            tool_id="capture_screen",
            name="capture_screen",
            description=(
                "Read-only desktop visual and compositor observation: captures the active monitor, "
                "specific window, or region via grim and returns exact window coordinates, monitor bounds, "
                "pixel-change hash/diff, and visual image context."
            ),
            tier=3,
            is_core=False,
            requires_bridge=False,
            requires_window=True,
            token_cost=220,
        )
        self.parameters = [
            ToolParameter(
                name="target",
                type="string",
                description="Capture scope: 'active_monitor' (default), 'active_window', 'window', 'monitor', 'region', or 'full'.",
                required=False,
                default="active_monitor",
            ),
            ToolParameter(
                name="window",
                type="string",
                description="Optional window title, class, or address to capture when target='window'.",
                required=False,
            ),
            ToolParameter(
                name="monitor",
                type="string",
                description="Optional monitor connector name (e.g. 'DP-1') when target='monitor'.",
                required=False,
            ),
            ToolParameter(
                name="region",
                type="string",
                description="Optional desktop region 'x,y widthxheight' when target='region'.",
                required=False,
            ),
            ToolParameter(
                name="include_image",
                type="boolean",
                description="Keep True (default) to attach the screenshot image so visible text, buttons, and coordinates can be read/verified.",
                required=False,
                default=True,
            ),
        ]
        self.ipc_client = ipc_client

    async def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        """Execute read-only desktop screenshot observation."""
        merged = dict(params or {})
        merged.update(kwargs)
        obs = await capture_desktop_observation(
            target=str(merged.get("target") or "active_monitor"),
            window=merged.get("window"),
            monitor=merged.get("monitor"),
            region=merged.get("region"),
            include_image=bool(merged.get("include_image", True)),
            max_edge=int(merged.get("max_edge") or 960),
            ipc_client=self.ipc_client,
        )
        vision_b64 = obs.pop("_vision_b64", None)
        if not obs.get("success"):
            return ToolResult(False, error=obs.get("error", "Screenshot capture failed"), output=obs)
        meta = {"_vision_b64": vision_b64} if vision_b64 else {}
        return ToolResult(True, output=obs, metadata=meta)


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "capture_screen",
        "description": (
            "Read-only desktop screenshot & layout observation. Captures active monitor, "
            "specific window, or full desktop and reports exact window/monitor coordinates and visual state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Capture scope: 'active_monitor', 'active_window', 'window', 'monitor', 'region', or 'full'.",
                    "default": "active_monitor",
                },
                "window": {"type": "string", "description": "Window title/class/address when target='window'."},
                "monitor": {"type": "string", "description": "Monitor connector name when target='monitor'."},
                "region": {"type": "string", "description": "Region 'x,y wxh' when target='region'."},
                "include_image": {"type": "boolean", "description": "Attach bounded visual image (default true).", "default": True},
            },
        },
    },
}
TIER = 3
IS_CORE = False
REQUIRES_BRIDGE = False
REQUIRES_WINDOW = True
REQUIRED_RING = 1
TUI_HINT = "📸 Observing desktop screen..."


__all__ = [
    "CaptureScreenTool",
    "capture_screen",
    "capture_desktop_observation",
    "capture_desktop_observation_sync",
    "TOOL_SCHEMA",
    "TIER",
    "IS_CORE",
    "REQUIRES_BRIDGE",
    "REQUIRES_WINDOW",
    "REQUIRED_RING",
]

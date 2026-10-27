"""Zelyr Vision — Vimium-style Set-of-Mark (SoM) Manager.

Replaces coordinate-regression clicking with a deterministic two-letter tag
system (AA, AB, … ZZ).  The pipeline is:

  1. Capture screen  →  VisionPipeline / os_vision
  2. SoMManager.generate_overlay(image)
       - OpenCV contour detection finds UI elements (buttons, panels, inputs)
       - Each element gets a unique 2-letter tag drawn on it in high-contrast text
       - tag → (center_x, center_y) stored in self.active_tags
  3. Annotated image is given to the VLM so it can read the tags
  4. VLM emits  click_tag(tag="AB")
  5. SomClickTool looks up (x, y) from active_tags and fires pyautogui.click()

No coordinate guessing.  No pixel regression.  Pure deterministic lookup.
"""

import itertools
import logging
import string
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional heavy deps (graceful degradation if headless/CI) ────────────────
try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False
    logger.warning("SoMManager: opencv-python not installed — overlay disabled.")

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    _PIL_AVAILABLE = True
except ImportError:
    Image = ImageDraw = ImageFont = None  # type: ignore
    _PIL_AVAILABLE = False
    logger.warning("SoMManager: Pillow not installed — overlay disabled.")

# ── Overlay visual constants ─────────────────────────────────────────────────
# Colors chosen for maximum legibility by small VLMs on any background.
TAG_BG_COLOR    = (15, 15, 15)       # near-black rectangle background (BGR)
TAG_TEXT_COLOR  = (0, 255, 128)      # bright green text  (BGR) — pops on dark bg
TAG_BORDER_COLOR = (0, 200, 80)      # slightly dimmer green border  (BGR)
TAG_BG_ALPHA    = 0.85               # overlay rectangle opacity

# Contour filtering thresholds (tune for your screen resolution)
MIN_CONTOUR_AREA  = 400              # px² — ignore single-pixel noise
MAX_CONTOUR_AREA_FRACTION = 0.35     # ignore regions > 35% of screen area
MIN_ASPECT_RATIO  = 0.05             # very thin horizontal lines → not a button
MAX_ASPECT_RATIO  = 20.0             # very thin vertical lines → not a button
FONT_SCALE        = 0.55             # cv2 font scale for tag labels
FONT_THICKNESS    = 2                # cv2 font thickness
PAD               = 5                # rectangle padding around text (px)


# ── Tag generator ─────────────────────────────────────────────────────────────

def _two_letter_tags() -> Generator[str, None, None]:
    """Yield AA, AB, AC … AZ, BA, BB … ZZ (676 unique tags)."""
    letters = string.ascii_uppercase
    for first, second in itertools.product(letters, letters):
        yield first + second


# ── SoMManager ───────────────────────────────────────────────────────────────

class SoMManager:
    """Vimium-style Set-of-Mark overlay manager for Zelyr desktop vision.

    Usage
    -----
    ::

        from zelyr.vision.som_manager import SoMManager

        manager = SoMManager()
        annotated_path = manager.generate_overlay("/tmp/zelyr_vision.png")
        # active_tags is now populated: {"AA": (412, 308), "AB": (800, 150), …}
        # Pass annotated_path to the VLM so it can read the tags.
        # When the VLM calls click_tag("AB"), look up (800, 150) and click.
    """

    def __init__(self) -> None:
        # Maps tag string → (center_x, center_y) in screen pixel coordinates.
        self.active_tags: Dict[str, Tuple[int, int]] = {}

        # Private generator state — reset with clear_tags().
        self._tag_gen: Generator[str, None, None] = _two_letter_tags()

        # Whether the underlying CV stack is available.
        self.is_available: bool = _CV2_AVAILABLE and _PIL_AVAILABLE

    # ── Public API ────────────────────────────────────────────────────────────

    def clear_tags(self) -> None:
        """Reset tag mapping and generator for a fresh screenshot cycle.

        Call this *before* each new generate_overlay() invocation so stale
        tags from a previous frame do not pollute the current lookup table.
        """
        self.active_tags.clear()
        self._tag_gen = _two_letter_tags()
        logger.debug("SoMManager: tag table cleared.")

    def generate_overlay(
        self,
        image_source: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
        *,
        return_array: bool = False,
    ) -> Optional[str]:
        """Detect UI elements, draw two-letter tags, persist the annotated image.

        Parameters
        ----------
        image_source:
            Either a filesystem path (str / Path) to a PNG/JPEG screenshot,
            or a BGR numpy array already in memory (e.g. from cv2.imread).
        output_path:
            Where to save the annotated image.  Defaults to
            ``/tmp/zelyr_som_overlay.png``.
        return_array:
            If True, also return the annotated numpy array instead of just
            the path string.  Used for unit testing without disk I/O.

        Returns
        -------
        str | None
            Absolute path to the saved annotated image, or None on failure.
        """
        if not self.is_available:
            logger.error("SoMManager.generate_overlay: cv2 or Pillow missing.")
            return None

        # ── 1. Load image ─────────────────────────────────────────────────────
        bgr = self._load_image(image_source)
        if bgr is None:
            return None

        screen_h, screen_w = bgr.shape[:2]
        max_area = screen_h * screen_w * MAX_CONTOUR_AREA_FRACTION

        # ── 2. Detect UI element contours ─────────────────────────────────────
        contours = self._detect_contours(bgr)

        # ── 3. Filter, deduplicate, assign tags, draw ──────────────────────────
        # Work on a copy so the original frame is unmodified.
        annotated = bgr.copy()
        seen_centers: list[Tuple[int, int]] = []  # used for overlap suppression

        for contour in contours:
            area = cv2.contourArea(contour)

            # ── Size filter ───────────────────────────────────────────────────
            if area < MIN_CONTOUR_AREA or area > max_area:
                continue

            # ── Bounding box + aspect ratio filter ───────────────────────────
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / h if h > 0 else 0
            if not (MIN_ASPECT_RATIO < aspect < MAX_ASPECT_RATIO):
                continue

            # ── Centroid via image moments ────────────────────────────────────
            M = cv2.moments(contour)
            if M["m00"] == 0:
                # Degenerate contour — fall back to bounding-box centre.
                cx, cy = x + w // 2, y + h // 2
            else:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

            # ── Minimum-distance deduplication ────────────────────────────────
            # Suppress tags that are very close to an already-tagged element,
            # which prevents stacking 20 tags on a single complex widget.
            if self._too_close(cx, cy, seen_centers, min_dist=40):
                continue
            seen_centers.append((cx, cy))

            # ── Assign tag ────────────────────────────────────────────────────
            try:
                tag = next(self._tag_gen)
            except StopIteration:
                # 676 tags exhausted — extremely unlikely on a single screen.
                logger.warning("SoMManager: ran out of tags (>676 elements).")
                break

            # Store the mapping (screen coordinates, not image-relative).
            self.active_tags[tag] = (cx, cy)

            # ── Draw tag badge ────────────────────────────────────────────────
            self._draw_tag(annotated, tag, cx, cy)

        logger.info(
            "SoMManager: tagged %d UI elements on %dx%d screen.",
            len(self.active_tags), screen_w, screen_h,
        )

        # ── 4. Save annotated image ───────────────────────────────────────────
        if output_path is None:
            output_path = "/tmp/zelyr_som_overlay.png"

        out_str = str(output_path)
        success = cv2.imwrite(out_str, annotated)
        if not success:
            logger.error("SoMManager: cv2.imwrite failed for path: %s", out_str)
            return None

        if return_array:
            return annotated  # type: ignore[return-value]  # test helper

        return out_str

    def lookup(self, tag: str) -> Optional[Tuple[int, int]]:
        """Return the (x, y) centre for a tag, or None if unknown.

        The tag lookup is case-insensitive to be forgiving of VLM output.
        """
        return self.active_tags.get(tag.upper())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_image(
        self, source: Union[str, Path, np.ndarray]
    ) -> Optional[np.ndarray]:
        """Return a BGR numpy array from a path or passthrough array."""
        if isinstance(source, np.ndarray):
            return source.copy()

        path = str(source)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            logger.error("SoMManager: could not read image at '%s'.", path)
        return bgr

    def _detect_contours(self, bgr: np.ndarray) -> list:
        """Run the full OpenCV contour-detection pipeline.

        Pipeline:
          BGR → Grayscale → Gaussian Blur → Canny Edges
            → Morphological Closing (joins broken edges)
            → findContours (external, sorted by area desc)
        """
        # Grayscale
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Gaussian blur — removes high-frequency texture noise so edges come
        # from structural boundaries, not gradients inside icons/text.
        blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)

        # Canny edge detection.  The thresholds are permissive so we catch
        # low-contrast UI borders (e.g. light-on-white buttons).
        edges = cv2.Canny(blurred, threshold1=30, threshold2=90)

        # Morphological closing: dilate then erode.  This bridges small gaps
        # in broken rectangle outlines so FindContours sees complete shapes.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find external contours only (we don't need nested holes).
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort by area descending so large containers are processed first.
        # The deduplication pass will still suppress closely stacked tags.
        return sorted(contours, key=cv2.contourArea, reverse=True)

    @staticmethod
    def _too_close(
        cx: int,
        cy: int,
        seen: list,
        min_dist: int,
    ) -> bool:
        """Return True if (cx, cy) is within min_dist pixels of any seen centre."""
        for sx, sy in seen:
            if abs(cx - sx) < min_dist and abs(cy - sy) < min_dist:
                return True
        return False

    @staticmethod
    def _draw_tag(img: np.ndarray, tag: str, cx: int, cy: int) -> None:
        """Draw a high-contrast tag badge centred at (cx, cy) on img in-place.

        Visual design:
          ┌─────────┐   ← dark rectangle border (TAG_BORDER_COLOR)
          │  AA     │   ← near-black fill (TAG_BG_COLOR)
          │         │   ← bright-green text (TAG_TEXT_COLOR)
          └─────────┘

        The badge is intentionally small (≈ 30×16 px at 1080p) so it sits
        *inside* the element without covering it entirely.  Small VLMs can
        read two uppercase letters at this size reliably.
        """
        font      = cv2.FONT_HERSHEY_SIMPLEX
        scale     = FONT_SCALE
        thickness = FONT_THICKNESS

        # Measure text bounding box to size the rectangle dynamically.
        (text_w, text_h), baseline = cv2.getTextSize(tag, font, scale, thickness)

        # Rectangle corners — centred on the element centroid.
        rx1 = cx - text_w // 2 - PAD
        ry1 = cy - text_h // 2 - PAD
        rx2 = cx + text_w // 2 + PAD
        ry2 = cy + text_h // 2 + PAD + baseline

        # Clamp to image bounds so edge-of-screen elements don't clip.
        h, w = img.shape[:2]
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w - 1, rx2), min(h - 1, ry2)

        # Semi-transparent fill: blend rectangle region with solid colour.
        roi = img[ry1:ry2, rx1:rx2]
        solid = np.full_like(roi, TAG_BG_COLOR, dtype=np.uint8)
        cv2.addWeighted(solid, TAG_BG_ALPHA, roi, 1 - TAG_BG_ALPHA, 0, roi)
        img[ry1:ry2, rx1:rx2] = roi

        # Crisp border on top of the blended region.
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), TAG_BORDER_COLOR, thickness=1)

        # Text — positioned so the baseline sits comfortably inside the rect.
        text_x = cx - text_w // 2
        text_y = cy + text_h // 2
        cv2.putText(
            img, tag,
            (text_x, text_y),
            font, scale, TAG_TEXT_COLOR, thickness,
            lineType=cv2.LINE_AA,   # anti-aliased — much easier for VLMs to read
        )

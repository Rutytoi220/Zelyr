"""Vision Pipeline Substrate for Zelyr v2.

Provides rapid screen-grabbing capabilities augmented with a
Set-of-Mark (SoM) grid for precise spatial reasoning by Vision Language Models.
"""
import base64
from io import BytesIO
import logging
from typing import Optional, Any
try:
    import mss as _mss
    mss: Any = _mss
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:  # pragma: no cover
    logging.getLogger(__name__).warning(f'VisionPipeline core dependencies missing: {e}')  # pragma: no cover
    mss = None  # pragma: no cover
try:
    import pygetwindow as gw
except (ImportError, NotImplementedError, Exception) as e:
    logging.getLogger(__name__).debug(f'pygetwindow not available (Linux fallback enabled): {e}')
    gw = None
logger = logging.getLogger(__name__)

class VisionPipeline:
    """Auto-generated docstring.

"""

    def __init__(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        self.is_available = mss is not None  # pragma: no cover

    def _apply_som_grid(self, img: 'Image.Image', grid_size: int=4) -> 'Image.Image':
        """Overlay a Set-of-Mark (SoM) coordinate grid onto the image."""
        if not self.is_available:  # pragma: no cover
            return img  # pragma: no cover
        draw = ImageDraw.Draw(img)  # pragma: no cover
        width, height = img.size  # pragma: no cover
        cell_width = width / grid_size  # pragma: no cover
        cell_height = height / grid_size  # pragma: no cover
        for i in range(1, grid_size):  # pragma: no cover
            x = int(i * cell_width)  # pragma: no cover
            draw.line([(x, 0), (x, height)], fill=(255, 0, 0, 128), width=2)  # pragma: no cover
            y = int(i * cell_height)  # pragma: no cover
            draw.line([(0, y), (width, y)], fill=(255, 0, 0, 128), width=2)  # pragma: no cover
        cols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # pragma: no cover
        try:  # pragma: no cover
            font = ImageFont.load_default()  # pragma: no cover
        except:  # pragma: no cover
            font = None  # pragma: no cover
        for row in range(grid_size):  # pragma: no cover
            for col in range(grid_size):  # pragma: no cover
                label = f'{cols[col]}{row + 1}'  # pragma: no cover
                x_center = int(col * cell_width + cell_width / 2)  # pragma: no cover
                y_center = int(row * cell_height + cell_height / 2)  # pragma: no cover
                try:  # pragma: no cover
                    bbox = draw.textbbox((0, 0), label, font=font)  # pragma: no cover
                    tw = bbox[2] - bbox[0]  # pragma: no cover
                    th = bbox[3] - bbox[1]  # pragma: no cover
                except AttributeError:  # pragma: no cover
                    tw, th = (20, 15)  # pragma: no cover
                rect_x1 = x_center - tw / 2 - 2  # pragma: no cover
                rect_y1 = y_center - th / 2 - 2  # pragma: no cover
                rect_x2 = x_center + tw / 2 + 2  # pragma: no cover
                rect_y2 = y_center + th / 2 + 2  # pragma: no cover
                draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=(0, 0, 0, 180))  # pragma: no cover
                draw.text((x_center - tw / 2, y_center - th / 2), label, fill=(255, 255, 255), font=font)  # pragma: no cover
        return img  # pragma: no cover

    def capture_active_window(self, with_grid: bool=True) -> Optional[str]:
        """Capture active window or primary monitor, apply grid, compress and return base64."""
        if not self.is_available:  # pragma: no cover
            logger.error('VisionPipeline dependencies not met.')  # pragma: no cover
            return None  # pragma: no cover
        monitor = None  # pragma: no cover
        try:  # pragma: no cover
            if gw is not None:  # pragma: no cover
                active_win = gw.getActiveWindow()  # pragma: no cover
                if active_win is not None:  # pragma: no cover
                    monitor = {'top': int(active_win.top), 'left': int(active_win.left), 'width': int(active_win.width), 'height': int(active_win.height)}  # pragma: no cover
        except (ImportError, NotImplementedError, Exception) as e:  # pragma: no cover
            logger.debug(f'Failed to get active window: {e}')  # pragma: no cover
        try:  # pragma: no cover
            with mss.mss() as sct:  # pragma: no cover
                if monitor is None:  # pragma: no cover
                    monitor = sct.monitors[1]  # pragma: no cover
                if monitor['width'] <= 0 or monitor['height'] <= 0:  # pragma: no cover
                    monitor = sct.monitors[1]  # pragma: no cover
                sct_img = sct.grab(monitor)  # pragma: no cover
                img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')  # pragma: no cover
                if with_grid:  # pragma: no cover
                    img = self._apply_som_grid(img)  # pragma: no cover
                buffered = BytesIO()  # pragma: no cover
                img.save(buffered, format='JPEG', quality=85)  # pragma: no cover
                return base64.b64encode(buffered.getvalue()).decode('utf-8')  # pragma: no cover
        except Exception as e:  # pragma: no cover
            logger.error(f'Screen capture failed: {e}')  # pragma: no cover
            return None  # pragma: no cover

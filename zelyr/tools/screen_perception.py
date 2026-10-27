import subprocess
import base64
import os
import shutil
import tempfile
import io
import logging
from PIL import Image
from zelyr.sdk.plugin import tool

logger = logging.getLogger(__name__)

@tool(
    name="capture_screen",
    description="Captures the current desktop screen and returns it as a Base64 encoded string. Crucial for visual UI analysis before taking physical mouse actions."
)
def capture_screen() -> str:
    """
    Captures the current desktop screen and returns it as a Base64 encoded string.
    Crucial for visual UI analysis before taking physical mouse actions.
    """
    try:
        temp_dir = tempfile.gettempdir()
        screenshot_path = os.path.join(temp_dir, "zelyr_vision_buffer.png")
        
        logger.info("Triggering screen capture...")
        if shutil.which("spectacle"):
            logger.info("Triggering spectacle screenshot...")
            subprocess.run(['spectacle', '-b', '-n', '-o', screenshot_path], check=True, timeout=3.0)
        elif shutil.which("grim"):
            logger.info("Triggering grim screenshot...")
            subprocess.run(['grim', '-c', screenshot_path], check=True, timeout=3.0)
        else:
            return "Failed to capture screen. Neither 'spectacle' nor 'grim' is installed."
        
        logger.info("Screenshot captured. Resizing image...")
        
        with open(screenshot_path, "rb") as image_file:
            raw_bytes = image_file.read()
            
        # Load into PIL and convert to RGB
        img = Image.open(io.BytesIO(raw_bytes))
        img = img.convert("RGB")
        
        # Downscale to max 1024 longest edge
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
        
        # Compress to JPEG
        out_buffer = io.BytesIO()
        img.save(out_buffer, format="JPEG", quality=85)
        jpeg_bytes = out_buffer.getvalue()
        
        encoded_string = base64.b64encode(jpeg_bytes).decode('utf-8')
            
        os.remove(screenshot_path)
        
        logger.info("VLM response received.")
        return f"SCREENSHOT_BASE64:{encoded_string}"
        
    except subprocess.TimeoutExpired:
        return "Failed to capture screen: The screenshot utility hung and timed out after 3 seconds."
    except subprocess.CalledProcessError as e:
        return f"Failed to capture screen due to subprocess error: {e}"
    except Exception as e:
        return f"Unexpected error during screen capture: {e}"

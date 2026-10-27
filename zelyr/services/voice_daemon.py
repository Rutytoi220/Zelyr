import logging
import asyncio
import threading
try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

logger = logging.getLogger(__name__)

class LocalVoicePipeline:
    """Manages local TTS (Text-to-Speech) and integrates STT."""
    
    def __init__(self):
        self.is_active = True
        self.engine = None
        if HAS_PYTTSX3:
            try:
                self.engine = pyttsx3.init()
            except Exception as e:
                logger.warning(f"VoicePipeline: Failed to initialize pyttsx3: {e}")
                self.engine = None

    def speak(self, text: str):
        """Synthesize and play speech in a background thread."""
        if not self.is_active:
            return
            
        if not self.engine:
            logger.info(f"[Voice Mock Output]: {text}")
            return
            
        def _run_speak():
            try:
                # Some implementations of pyttsx3 require re-init per thread on Linux
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"VoicePipeline TTS error: {e}")
                
        thread = threading.Thread(target=_run_speak, daemon=True)
        thread.start()

    def set_active(self, active: bool):
        """Toggle TTS feedback on or off."""
        self.is_active = active
        status = "Active" if active else "Muted"
        logger.info(f"VoicePipeline: Set to {status}")

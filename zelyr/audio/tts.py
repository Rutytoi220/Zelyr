import pyttsx3
import asyncio
import re
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class TextToSpeechEngine:
    """Async wrapper for local Text-To-Speech (pyttsx3)."""
    
    _instance = None
    
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        try:
            self.engine = pyttsx3.init()
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            self.engine = None
        
    def _sanitize_text(self, text: str) -> str:
        """Strip markdown, code blocks, and XML tags for clean speech."""
        # Remove code blocks
        text = re.sub(r'```.*?```', ' [Code block omitted] ', text, flags=re.DOTALL)
        # Remove inline code
        text = re.sub(r'`.*?`', '', text)
        # Remove bold/italic markdown
        text = re.sub(r'[\*\_\#\>\[\]]', '', text)
        # Remove HTML/XML tags
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

    def _speak_sync(self, text: str):
        if not self.engine:
            return
        clean_text = self._sanitize_text(text)
        if clean_text:
            self.engine.say(clean_text)
            self.engine.runAndWait()

    async def speak(self, text: str):
        """Asynchronously speak the sanitized text without blocking the UI thread."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self._speak_sync, text)

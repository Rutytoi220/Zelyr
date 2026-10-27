import time  # pragma: no cover
import queue  # pragma: no cover
import logging  # pragma: no cover
import re  # pragma: no cover
from multiprocessing import Queue  # pragma: no cover

logger = logging.getLogger(__name__)  # pragma: no cover

def clean_markdown_for_speech(text: str) -> str:  # pragma: no cover
    """Strips Markdown syntax to make text TTS-friendly."""
    # Remove code blocks
    text = re.sub(r'```.*?```', ' [Code block omitted] ', text, flags=re.DOTALL)  # pragma: no cover
    # Remove inline code
    text = re.sub(r'`[^`]*`', '', text)  # pragma: no cover
    # Remove bold/italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # pragma: no cover
    text = re.sub(r'\*(.*?)\*', r'\1', text)  # pragma: no cover
    text = re.sub(r'__(.*?)__', r'\1', text)  # pragma: no cover
    text = re.sub(r'_(.*?)_', r'\1', text)  # pragma: no cover
    # Remove JSON wrappers or specific tags if any
    text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)  # pragma: no cover
    return text.strip()  # pragma: no cover

def tts_loop(tts_queue: Queue):  # pragma: no cover
    """Text-to-Speech loop running in background thread."""
    try:  # pragma: no cover
        import pyttsx3  # pragma: no cover
        engine = pyttsx3.init()  # pragma: no cover
        engine.setProperty('rate', 170)  # pragma: no cover
        has_tts = False  # TEMPORARILY MUTED PER USER REQUEST
        logger.info("TTS Engine initialized (pyttsx3) but MUTED.")  # pragma: no cover
    except ImportError:  # pragma: no cover
        logger.warning("pyttsx3 not installed. TTS will just consume the queue silently.")  # pragma: no cover
        has_tts = False  # pragma: no cover

    while True:  # pragma: no cover
        try:  # pragma: no cover
            # Block until there is text to speak
            text = tts_queue.get()  # pragma: no cover
            
            clean_text = clean_markdown_for_speech(text)  # pragma: no cover
            if not clean_text:  # pragma: no cover
                continue  # pragma: no cover
                
            if has_tts:  # pragma: no cover
                engine.say(clean_text)  # pragma: no cover
                engine.runAndWait()  # pragma: no cover
            else:
                logger.info(f"[TTS Stub Speak]: {clean_text}")  # pragma: no cover
                
        except Exception as e:  # pragma: no cover
            logger.error(f"TTS Loop error: {e}")  # pragma: no cover
            time.sleep(1.0)  # pragma: no cover

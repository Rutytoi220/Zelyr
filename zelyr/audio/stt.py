import numpy as np
import sounddevice as sd
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """Local STT using faster-whisper."""
    _instance = None
    
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        try:
            from faster_whisper import WhisperModel
            # Load small/tiny model for fast local transcription
            self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            logger.info("WhisperModel loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load WhisperModel: {e}")
            self.model = None
            
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _transcribe_sync(self, audio_data: np.ndarray) -> str:
        if not self.model or len(audio_data) == 0:
            return ""
        
        # faster-whisper expects float32 array at 16000 Hz
        segments, info = self.model.transcribe(audio_data, beam_size=1)
        text = "".join([segment.text for segment in segments])
        return text.strip()

    async def transcribe(self, audio_data: np.ndarray) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._transcribe_sync, audio_data)


class AudioRecorder:
    """Manages Push-to-Talk audio recording."""
    def __init__(self, samplerate=16000):
        self.samplerate = samplerate
        self.recording = False
        self.frames = []
        self.stream = None

    def start_recording(self):
        self.frames = []
        self.recording = True
        
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio status: {status}")
            if self.recording:
                self.frames.append(indata.copy())

        try:
            self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, dtype='float32', callback=callback)
            self.stream.start()
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")

    def stop_recording(self) -> np.ndarray:
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if not self.frames:
            return np.array([], dtype=np.float32)
            
        return np.concatenate(self.frames, axis=0).flatten()


class WakeWordDaemon(QObject):
    """Background listener for Wake Word mode."""
    wake_word_detected = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._thread = None
        
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            
    def _listen_loop(self):
        try:
            import openwakeword
            from openwakeword.model import Model
            # Using 'hey_jarvis' as a stand-in since 'hey_zelyr' isn't pre-trained
            oww_model = Model(wakeword_models=["hey_jarvis"])
            
            # Simple 16kHz audio capture chunk loop
            with sd.InputStream(samplerate=16000, channels=1, dtype='int16') as stream:
                while self._running:
                    # Read 1280 frames (~80ms)
                    data, overflow = stream.read(1280)
                    if overflow:
                        continue
                        
                    # Feed to openwakeword
                    prediction = oww_model.predict(data.flatten())
                    
                    for mdl, score in prediction.items():
                        if score > 0.5:  # Threshold
                            self.wake_word_detected.emit()
                            # Reset state to prevent multiple immediate triggers
                            oww_model.reset()
                            
        except ImportError:
            logger.error("openwakeword not installed. Wake Word mode unavailable.")
        except Exception as e:
            logger.error(f"WakeWord daemon error: {e}")

import logging
import threading
import time
from zelyr.core.events import EventBus

logger = logging.getLogger(__name__)

class WakeWordService:
    """Isolated background thread for openwakeword."""
    
    def __init__(self, event_bus: EventBus):
        self._bus = event_bus
        self._running = False
        self._paused = False
        self._thread = None
        
        # Subscribe to resume/pause events
        self._bus.subscribe("audio.wakeword.resume", self.resume)
        self._bus.subscribe("audio.wakeword.pause", self.pause)
        self._bus.subscribe("config.updated", self._on_config_updated)
        
    def _on_config_updated(self, event):
        from zelyr.config import get_config
        enabled = getattr(get_config(), "wake_word_enabled", False)
        if enabled and not self._running:
            self.start()
        elif not enabled and self._running:
            self.stop()
        
    def start(self):
        if self._running:
            return
        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("WakeWordService background thread started.")
        
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            
    def pause(self, *args, **kwargs):
        self._paused = True
        
    def resume(self, *args, **kwargs):
        self._paused = False
        
    def _listen_loop(self):
        try:
            import openwakeword
            from openwakeword.model import Model
            import sounddevice as sd
            import numpy as np
            
            # Download/load model
            oww_model = Model(wakeword_models=["hey_jarvis"])
            logger.info("WakeWord model 'hey_jarvis' loaded.")
            
            while self._running:
                if self._paused:
                    time.sleep(0.1) # Aggressively yield CPU while paused
                    continue
                    
                # We open and close the stream inside the loop so we can yield ALSA lock
                try:
                    with sd.InputStream(samplerate=16000, channels=1, dtype='int16') as stream:
                        while self._running and not self._paused:
                            # 1280 frames = 80ms chunk
                            data, overflow = stream.read(1280)
                            if overflow:
                                logger.debug("Audio overflow in WakeWordService")
                                
                            # Convert to numpy and predict
                            prediction = oww_model.predict(data.flatten())
                            
                            for mdl, score in prediction.items():
                                if score > 0.5:
                                    logger.info(f"Wake word detected! Score: {score}")
                                    self._paused = True # Auto-pause to release ALSA
                                    oww_model.reset()
                                    self._bus.publish_sync("audio.wakeword.detected", {})
                                    break
                except Exception as e:
                    logger.error(f"WakeWord audio stream error: {e}")
                    time.sleep(1.0) # Backoff
                    
        except ImportError:
            logger.warning("openwakeword not installed. Wake Word mode disabled.")
        except Exception as e:
            logger.error(f"WakeWordService error: {e}")


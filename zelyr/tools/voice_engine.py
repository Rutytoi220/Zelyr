import logging
import asyncio
import queue
import tempfile
import os
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

class VoiceDictationEngine:
    def __init__(self, model_size="tiny.en"):
        self.model_size = model_size
        self.model = None
        self._is_recording = False
        self._audio_queue = queue.Queue()
        self._stream = None

    def _load_model(self):
        if not self.model:
            from faster_whisper import WhisperModel
            import torch
            
            # Check for CUDA
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            logger.info(f"Loading faster-whisper model '{self.model_size}' on {device}")
            self.model = WhisperModel(self.model_size, device=device, compute_type=compute_type)

    def start_recording(self):
        import sounddevice as sd
        
        self._is_recording = True
        self._audio_queue = queue.Queue()
        
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            self._audio_queue.put(indata.copy())
            
        self._stream = sd.InputStream(samplerate=16000, channels=1, callback=callback, dtype='float32')
        self._stream.start()
        logger.info("Started voice recording.")

    def stop_recording_and_transcribe(self) -> str:
        if not self._is_recording:
            return ""
            
        self._is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            
        logger.info("Stopped voice recording. Processing audio...")
        
        audio_data = []
        while not self._audio_queue.empty():
            audio_data.append(self._audio_queue.get())
            
        if not audio_data:
            return ""
            
        # Concatenate all chunks
        audio_np = np.concatenate(audio_data, axis=0)
        
        # Save to temp file since faster-whisper expects a file path or waveform array
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            
        sf.write(temp_path, audio_np, 16000)
        
        try:
            self._load_model()
            segments, info = self.model.transcribe(temp_path, beam_size=5)
            text = " ".join([segment.text for segment in segments]).strip()
            return text
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return ""
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

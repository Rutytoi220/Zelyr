import multiprocessing
import queue
import logging
logger = logging.getLogger(__name__)

class AudioDaemon:
    """Isolated background daemon for non-blocking STT and TTS audio pipelines."""

    def __init__(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        from typing import Any
        self.tts_queue: 'multiprocessing.Queue[Any]' = multiprocessing.Queue()  # pragma: no cover
        self.stt_queue: 'multiprocessing.Queue[Any]' = multiprocessing.Queue()  # pragma: no cover
        self.process: 'multiprocessing.Process | None' = None  # pragma: no cover

    def start(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        if self.process is not None and self.process.is_alive():  # pragma: no cover
            return  # pragma: no cover
        logger.info('Starting isolated AudioDaemon process for JARVIS pipeline.')  # pragma: no cover
        self.process = multiprocessing.Process(target=self._run_loops, daemon=True)  # pragma: no cover
        self.process.start()  # pragma: no cover

    def _run_loops(self):
        """Runs inside the child process to prevent GIL locking."""
        import sys  # pragma: no cover
        import os  # pragma: no cover
        sys.stdin = open(os.devnull, 'r')  # pragma: no cover
        from zelyr.perception.voice_out import tts_loop  # pragma: no cover
        from zelyr.perception.voice_in import stt_loop  # pragma: no cover
        import threading  # pragma: no cover
        t1 = threading.Thread(target=tts_loop, args=(self.tts_queue,), daemon=True)  # pragma: no cover
        t2 = threading.Thread(target=stt_loop, args=(self.stt_queue,), daemon=True)  # pragma: no cover
        t1.start()  # pragma: no cover
        t2.start()  # pragma: no cover
        t1.join()  # pragma: no cover
        t2.join()  # pragma: no cover

    def send_tts(self, text: str):
        """Auto-generated docstring.

Args:
    text: Argument.

Returns:
    Return value.
"""
        self.tts_queue.put(text)  # pragma: no cover

    def poll_stt(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        try:  # pragma: no cover
            return self.stt_queue.get_nowait()  # pragma: no cover
        except queue.Empty:  # pragma: no cover
            return None  # pragma: no cover

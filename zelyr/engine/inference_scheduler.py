"""Centralized GPU Inference Queue for Zelyr."""

import asyncio
import threading
import logging
from typing import Callable, Any
from dataclasses import dataclass, field
import contextlib

logger = logging.getLogger(__name__)

@dataclass(order=True)
class PriorityRequest:
    priority: int
    id: int = field(compare=False)
    event: threading.Event = field(compare=False)

class InferenceScheduler:
    """Singleton priority queue for Ollama GPU inference requests."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InferenceScheduler, cls).__new__(cls)
                cls._instance._init_once()
            return cls._instance

    def _init_once(self):
        self._task_id_counter = 0
        self._task_id_lock = threading.Lock()
        
        import queue
        self._queue = queue.PriorityQueue()
        self._active_lock = threading.Lock()
        
        self._worker_thread = threading.Thread(target=self._worker, daemon=True, name="InferenceQueueWorker")
        self._worker_thread.start()

    def _worker(self):
        logger.info("InferenceScheduler worker started.")
        while True:
            # Get the highest priority request
            req = self._queue.get()
            
            # Acquire the active lock (blocks if someone else is currently executing)
            self._active_lock.acquire()
            
            # Notify the waiting thread that it can proceed
            req.event.set()
            self._queue.task_done()

    @contextlib.contextmanager
    def priority_lock(self, priority: int):
        """Context manager to acquire the inference lock at a given priority."""
        with self._task_id_lock:
            self._task_id_counter += 1
            task_id = self._task_id_counter
            
        req_event = threading.Event()
        req = PriorityRequest(priority=priority, id=task_id, event=req_event)
        
        self._queue.put(req)
        
        # Block until the worker grants us the lock
        req_event.wait()
        
        try:
            yield
        finally:
            # Release the lock for the next item in the queue
            self._active_lock.release()

def get_scheduler() -> InferenceScheduler:
    return InferenceScheduler()

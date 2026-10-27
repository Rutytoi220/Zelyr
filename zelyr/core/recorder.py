"""Flight Recorder for zero-dependency telemetry tracing."""
from zelyr.compat import state_dir
import json
import logging
import os
import queue
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
from zelyr.core.events import EventBus, Event
logger = logging.getLogger(__name__)
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

class FlightRecorder:
    """Non-blocking JSONL telemetry recorder."""

    def __init__(self, bus: EventBus, trace_dir: Optional[str]=None):
        """Auto-generated docstring.

Args:
    bus: Argument.
    trace_dir: Argument.

Returns:
    Return value.
"""
        self.bus = bus
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        if not trace_dir:
            trace_dir = str(state_dir() / 'traces')
        self.trace_dir = Path(trace_dir)
        self.trace_file = self.trace_dir / 'flight_recorder.jsonl'
        self._current_size = 0

    def start(self) -> None:
        """Start the recorder background thread."""
        if self._running:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        if self.trace_file.exists():
            self._current_size = self.trace_file.stat().st_size
        self._running = True
        self._thread = threading.Thread(target=self._writer_loop, name='FlightRecorder', daemon=True)
        self._thread.start()
        self.bus.subscribe('bus.published', self._on_event)

    def stop(self) -> None:
        """Stop the recorder and flush the queue."""
        if not self._running:
            return
        self.bus.unsubscribe('bus.published', self._on_event)
        self._running = False
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=2.0)

    def _on_event(self, event: Event) -> None:
        """Enqueue meta-events for recording."""
        if not self._running:
            return
        orig_event = event.data.get('event')
        orig_payload = event.data.get('original_payload')
        from datetime import UTC
        trace_record = {'timestamp': datetime.now(UTC).isoformat().replace('+00:00', 'Z'), 'event_type': orig_event, 'data': self._scrub_secrets(orig_payload)}
        try:
            self._queue.put_nowait(trace_record)
        except queue.Full:
            pass

    def _writer_loop(self) -> None:
        """Background thread loop to write traces to disk."""
        while self._running or not self._queue.empty():
            try:
                record = self._queue.get(timeout=0.5)
                if record is None:
                    continue
                line = json.dumps(record, default=str) + '\n'
                line_bytes = line.encode('utf-8')
                self._rotate_if_needed(len(line_bytes))
                with open(self.trace_file, 'ab') as f:
                    f.write(line_bytes)
                self._current_size += len(line_bytes)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f'FlightRecorder write error: {e}')

    def _rotate_if_needed(self, next_write_size: int) -> None:
        """Rotate the log file if it exceeds MAX_BYTES."""
        if self._current_size + next_write_size <= MAX_BYTES:
            return
        if not self.trace_file.exists():
            self._current_size = 0
            return
        for i in range(BACKUP_COUNT - 1, 0, -1):
            src = self.trace_dir / f'flight_recorder.jsonl.{i}'
            dst = self.trace_dir / f'flight_recorder.jsonl.{i + 1}'
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
        dst1 = self.trace_dir / 'flight_recorder.jsonl.1'
        if dst1.exists():
            dst1.unlink()
        self.trace_file.rename(dst1)
        self._current_size = 0

    def _scrub_secrets(self, data: Any) -> Any:
        """Recursively scrub sensitive information from the data."""
        if isinstance(data, dict):
            return {k: self._scrub_secrets(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._scrub_secrets(v) for v in data]
        elif isinstance(data, str):
            s = re.sub('Bearer [a-zA-Z0-9\\-\\._~+/]+', 'Bearer ***', data)
            s = re.sub('sk-[a-zA-Z0-9]{20,}', 'sk-***', s)
            s = re.sub('(://[^:]+:)([^@]+)(@)', '\\1***\\3', s)
            try:
                home_dir = str(Path.home())
                if home_dir and home_dir != '/':
                    s = s.replace(home_dir, '~')
            except Exception:
                pass
            return s
        return data

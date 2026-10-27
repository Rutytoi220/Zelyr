"""Filesystem Watchdog Service for Auto-Indexing Semantic Memory."""
import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import List, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from zelyr.config import get_config
from zelyr.memory.vector_store import VectorMemoryEngine
from zelyr.tools.document_reader import ReadDocumentContentTool

logger = logging.getLogger(__name__)

class AutoIndexingEventHandler(FileSystemEventHandler):
    """Handles file creation and modification events for auto-indexing."""
    
    SUPPORTED_EXTENSIONS = {'.md', '.txt', '.py', '.json', '.pdf'}
    
    def __init__(self, memory_engine: VectorMemoryEngine, loop: asyncio.AbstractEventLoop):
        self.memory_engine = memory_engine
        self.loop = loop
        self.doc_reader = ReadDocumentContentTool()
        self._pending_files: dict[str, float] = {}
        self._debounce_seconds = 2.0
        self._lock = threading.Lock()
        
    def _is_supported(self, path: str) -> bool:
        return Path(path).suffix.lower() in self.SUPPORTED_EXTENSIONS
        
    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and self._is_supported(event.src_path):
            self._schedule_index(event.src_path)
            
    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and self._is_supported(event.src_path):
            self._schedule_index(event.src_path)

    def _schedule_index(self, file_path: str):
        """Debounce events for a file path."""
        with self._lock:
            self._pending_files[file_path] = time.time()
            
    def process_pending(self):
        """Called periodically to process debounced files."""
        now = time.time()
        to_process = []
        with self._lock:
            for path, timestamp in list(self._pending_files.items()):
                if now - timestamp >= self._debounce_seconds:
                    to_process.append(path)
                    del self._pending_files[path]
                    
        for path_str in to_process:
            path = Path(path_str)
            if not path.exists():
                continue
                
            try:
                # Extract text using existing document reader tool logic
                text = self.doc_reader._extract_text(path)
                if not text or len(text.strip()) == 0:
                    continue
                    
                # Index into vector memory asynchronously
                logger.info(f"[Watchdog] Auto-indexing file: {path.name}")
                asyncio.run_coroutine_threadsafe(
                    self.memory_engine.add_document(
                        doc_id=str(path.absolute()),
                        text=text,
                        metadata={"filename": path.name, "path": str(path.absolute()), "source": "watchdog"}
                    ),
                    self.loop
                )
            except Exception as e:
                logger.error(f"[Watchdog] Failed to index {path_str}: {e}")

class DirectoryWatchdog:
    """Background daemon to watch directories and auto-index files."""
    
    def __init__(self, memory_engine: Optional[VectorMemoryEngine] = None):
        self.config = get_config()
        self.memory_engine = memory_engine or VectorMemoryEngine()
        self.observer = None
        self._running = False
        self._loop = None
        self._worker_thread = None
        
    def start(self, loop: asyncio.AbstractEventLoop):
        if not self.config.auto_index_watchdog:
            logger.info("Auto-indexing watchdog is disabled in config.")
            return
            
        if not self.config.monitored_paths:
            logger.warning("Auto-indexing watchdog enabled but no monitored_paths configured.")
            return
            
        if self._running:
            return
            
        self._loop = loop
        self.handler = AutoIndexingEventHandler(self.memory_engine, self._loop)
        self.observer = Observer()
        
        for path_str in self.config.monitored_paths:
            path = Path(path_str).expanduser().resolve()
            if path.exists() and path.is_dir():
                self.observer.schedule(self.handler, str(path), recursive=True)
                logger.info(f"Watchdog monitoring directory: {path}")
            else:
                logger.warning(f"Watchdog skip missing directory: {path}")
                
        self.observer.start()
        self._running = True
        
        self._worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._worker_thread.start()
        logger.info("DirectoryWatchdog started.")
        
    def _process_loop(self):
        while self._running:
            try:
                self.handler.process_pending()
            except Exception as e:
                logger.error(f"Watchdog process error: {e}")
            time.sleep(0.5)
            
    def stop(self):
        self._running = False
        if self.observer:
            self.observer.stop()
            self.observer.join()
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
        logger.info("DirectoryWatchdog stopped.")

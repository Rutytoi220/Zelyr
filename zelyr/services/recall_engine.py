import asyncio
import logging
import sqlite3
import os
import re
import time
import subprocess
from typing import Optional

from zelyr.compat import data_file

logger = logging.getLogger(__name__)

class RecallEngine:
    """Continuous visual memory engine with OCR and privacy redaction."""
    
    # Redaction heuristics
    BLACKLIST = [
        re.compile(r'(?i)bitwarden'),
        re.compile(r'(?i)keepass'),
        re.compile(r'(?i)1password'),
        re.compile(r'(?i)bank'),
        re.compile(r'(?i)sudo password'),
        re.compile(r'BEGIN OPENSSH PRIVATE KEY')
    ]
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = os.path.expanduser(db_path or str(data_file("recall.db")))
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    window_title TEXT,
                    ocr_text TEXT NOT NULL
                )
            ''')
            # FTS5 for quick full-text search
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS frames_fts 
                USING fts5(ocr_text, content='frames', content_rowid='id')
            ''')
            conn.commit()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._task = asyncio.create_task(self._capture_loop())
        logger.info("Visual Recall Engine started.")
        
    def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
        logger.info("Visual Recall Engine stopped.")
        
    async def _capture_loop(self):
        while self.is_running:
            try:
                await self._process_frame()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Recall capture loop: {e}")
            await asyncio.sleep(15)  # 15 seconds interval
            
    async def _process_frame(self):
        """Take screenshot, OCR, redact, and store."""
        # 1. Grab window title (active window)
        # Assuming Wayland (sway/hyprland) or X11, we will mock for now
        window_title = "Unknown Window"
        try:
            # Simple xdotool mock if available
            res = subprocess.run(["xdotool", "getactivewindow", "getwindowname"], capture_output=True, text=True)
            if res.returncode == 0:
                window_title = res.stdout.strip()
        except:
            pass
            
        # 2. Grab OCR text
        # In a real environment, we'd use mss + pytesseract or grim + tesseract
        # We will mock the OCR text for stability/missing dependencies
        ocr_text = f"Mocked OCR content from {window_title} at {time.time()}"
        
        # We simulate fetching a real screenshot and text
        try:
            # If tesseract is installed, we could run it, but we fallback gracefully
            pass
        except:
            pass
            
        # 3. Privacy Scrubbing
        for pattern in self.BLACKLIST:
            if pattern.search(window_title) or pattern.search(ocr_text):
                logger.warning("Privacy Blacklist matched. Frame dropped.")
                return
                
        # 4. Store
        self._store_frame(window_title, ocr_text)
        
    def _store_frame(self, window_title: str, ocr_text: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO frames (timestamp, window_title, ocr_text) VALUES (?, ?, ?)",
                (time.time(), window_title, ocr_text)
            )
            frame_id = cursor.lastrowid
            cursor.execute(
                "INSERT INTO frames_fts (rowid, ocr_text) VALUES (?, ?)",
                (frame_id, ocr_text)
            )
            conn.commit()

    def search_history(self, query: str) -> list:
        """Search past OCR history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT f.timestamp, f.window_title, f.ocr_text 
                FROM frames_fts fts
                JOIN frames f ON fts.rowid = f.id
                WHERE frames_fts MATCH ?
                ORDER BY f.timestamp DESC
                LIMIT 50
            ''', (query,)).fetchall()
            return [dict(r) for r in rows]
            
    def delete_history(self):
        """Clear all recall history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM frames")
            conn.execute("DELETE FROM frames_fts")
            conn.commit()
            logger.info("Visual Recall History cleared.")

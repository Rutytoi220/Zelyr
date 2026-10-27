import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from zelyr.compat import data_file

logger = logging.getLogger(__name__)

class AuditLedger:
    """SQLite-based ledger for auditing AI tool executions."""
    
    def __init__(self, db_path: Optional[Path] = None):
        if not db_path:
            db_path = data_file("audit.db")
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        agent_name TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        arguments TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        status TEXT NOT NULL
                    )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize audit DB: {e}")

    def log_execution(self, agent_name: str, tool_name: str, arguments: Dict[str, Any], risk_level: str, status: str) -> int:
        """Log a tool execution attempt."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO audit_log (timestamp, agent_name, tool_name, arguments, risk_level, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    agent_name,
                    tool_name,
                    json.dumps(arguments),
                    risk_level,
                    status
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to log to audit DB: {e}")
            return -1

    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent audit logs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM audit_log ORDER BY id DESC LIMIT ?', (limit,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve audit logs: {e}")
            return []

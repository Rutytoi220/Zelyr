from zelyr.compat import data_dir
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class TokenBudgetManager:
    """SQLite-based ledger for tracking and enforcing cloud LLM token budgets."""
    
    # Approx cost per 1k tokens for claude-3.5-sonnet
    # $3.00 / 1M input, $15.00 / 1M output
    COST_PER_1K_PROMPT = 0.003
    COST_PER_1K_COMPLETION = 0.015

    def __init__(self, db_path: Optional[Path] = None):
        if not db_path:
            db_path = data_dir() / 'token_budget.db'
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS usage_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        date TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        prompt_tokens INTEGER NOT NULL,
                        completion_tokens INTEGER NOT NULL,
                        estimated_cost_usd REAL NOT NULL
                    )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize budget DB: {e}")

    def log_usage(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> int:
        """Log API consumption and calculate cost."""
        cost = (prompt_tokens / 1000.0) * self.COST_PER_1K_PROMPT + \
               (completion_tokens / 1000.0) * self.COST_PER_1K_COMPLETION
               
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now()
                cursor.execute('''
                    INSERT INTO usage_log (timestamp, date, provider, model, prompt_tokens, completion_tokens, estimated_cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    now.isoformat(),
                    now.date().isoformat(),
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cost
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to log to budget DB: {e}")
            return -1

    def get_today_usage(self) -> Dict[str, Any]:
        """Get token and cost usage for today."""
        today = date.today().isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT SUM(prompt_tokens + completion_tokens) as total_tokens,
                           SUM(estimated_cost_usd) as total_cost
                    FROM usage_log
                    WHERE date = ?
                ''', (today,))
                row = cursor.fetchone()
                return {
                    'total_tokens': row[0] or 0,
                    'total_cost': row[1] or 0.0
                }
        except Exception as e:
            logger.error(f"Failed to get today usage: {e}")
            return {'total_tokens': 0, 'total_cost': 0.0}

    def can_afford_cloud_call(self, estimated_tokens: int = 4000) -> Tuple[bool, str, float]:
        """Check if we have enough budget to make a cloud call.
        Returns: (is_allowed, reason, percent_used)
        """
        from zelyr.config import get_config
        config = get_config()
        
        # Use settings from config if available, otherwise use defaults
        daily_limit = getattr(config, 'daily_cloud_token_limit', 50000)
        
        usage = self.get_today_usage()
        total_tokens = usage['total_tokens']
        
        percent_used = (total_tokens / daily_limit) * 100 if daily_limit > 0 else 100.0
        
        if total_tokens + estimated_tokens > daily_limit:
            return False, f"[⚠️ Cloud Budget Exhausted ({total_tokens}/{daily_limit} tokens) — Executing via Local Ollama fallback]", percent_used
            
        if percent_used >= 90.0:
             return False, f"[⚠️ Cloud Budget >= 90% Exhausted ({percent_used:.1f}%) — Executing via Local Ollama fallback to prevent overage]", percent_used
             
        return True, "Within budget limits.", percent_used

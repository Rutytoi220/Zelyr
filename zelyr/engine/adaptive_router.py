from zelyr.compat import config_dir as _zelyr_config_dir
import asyncio
import aiosqlite
import aiohttp
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TelemetryDB:
    """Async database layer for storing and retrieving model telemetry."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            config_dir = _zelyr_config_dir()
            config_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(config_dir / "telemetry.db")
        else:
            self.db_path = db_path
            
    async def _init_db(self):
        """Initialize the database schema."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode = WAL;")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS router_metrics (
                    model_name TEXT, 
                    task_type TEXT, 
                    avg_latency REAL, 
                    success_score INTEGER, 
                    PRIMARY KEY (model_name, task_type)
                )
                """
            )
            await db.commit()

    async def update_metrics(self, model_name: str, task_type: str, latency: float, feedback_modifier: int):
        """Update the metrics for a given model and task type."""
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            # Upsert logic to handle rolling average and score
            cursor = await db.execute(
                "SELECT avg_latency, success_score FROM router_metrics WHERE model_name = ? AND task_type = ?",
                (model_name, task_type)
            )
            row = await cursor.fetchone()
            
            if row:
                current_latency, current_score = row
                # Simple moving average for latency
                new_latency = (current_latency + latency) / 2.0
                new_score = current_score + feedback_modifier
                await db.execute(
                    "UPDATE router_metrics SET avg_latency = ?, success_score = ? WHERE model_name = ? AND task_type = ?",
                    (new_latency, new_score, model_name, task_type)
                )
            else:
                await db.execute(
                    "INSERT INTO router_metrics (model_name, task_type, avg_latency, success_score) VALUES (?, ?, ?, ?)",
                    (model_name, task_type, latency, feedback_modifier)
                )
            await db.commit()

    async def get_context(self, task_type: str) -> str:
        """Retrieve a formatted string of the top 3 models and their stats for the requested task."""
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT model_name, avg_latency, success_score 
                FROM router_metrics 
                WHERE task_type = ? 
                ORDER BY success_score DESC, avg_latency ASC 
                LIMIT 3
                """,
                (task_type,)
            )
            rows = await cursor.fetchall()
            
            if not rows:
                return "No historical telemetry available for this task type."
                
            context_parts = []
            for row in rows:
                model_name, latency, score = row
                context_parts.append(f"{model_name} (Latency: {latency:.1f}s, Score: {score})")
                
            return ", ".join(context_parts)

class NeuralRouter:
    """LLM-driven routing engine that uses Qwen to dynamically route tasks."""
    
    def __init__(self, db: TelemetryDB = None, ollama_url: str = "http://127.0.0.1:11434"):
        self.db = db or TelemetryDB()
        self.ollama_url = ollama_url
        
    async def route(self, task_type: str, user_prompt: str, available_models: list[str]) -> str:
        """Dynamically route a task to the optimal model based on telemetry."""
        if not available_models:
            raise ValueError("No models available for routing.")
            
        db_context_string = await self.db.get_context(task_type)
        
        system_prompt = (
            "You are the Zelyr routing kernel. You receive a task and hardware telemetry. "
            "You must output ONLY the exact string name of the best model from the available list. "
            "NEVER output conversational text, markdown, or explanations."
        )
        
        prompt = f"""Available: ['hermes:8b', 'gemma4:12b', 'coder:7b']
Task Type: coding
Telemetry: gemma4:12b (Latency: 12.5s, Score: 2), coder:7b (Latency: 2.1s, Score: 5)
Output: coder:7b

Available: {available_models}
Task Type: {task_type}
Telemetry: {db_context_string}
Output:"""

        payload = {
            "model": "qwen2.5:1.5b",
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ollama_url}/api/generate", json=payload, timeout=5.0) as response:
                    if response.status == 200:
                        data = await response.json()
                        raw_output = data.get("response", "")
                        
                        # Strip all whitespace and line breaks
                        clean_output = raw_output.strip().replace("\\n", "").replace("\\r", "")
                        
                        # Validate the output string exists in available_models
                        if clean_output in available_models:
                            return clean_output
                        else:
                            logger.error(f"[NeuralRouter] Qwen hallucinated invalid model: '{clean_output}'. Fallback triggered.")
                            return available_models[0]
                    else:
                        logger.error(f"[NeuralRouter] Ollama API returned status {response.status}. Fallback triggered.")
                        return available_models[0]
        except Exception as e:
            logger.error(f"[NeuralRouter] Exception during routing request: {e}. Fallback triggered.")
            return available_models[0]

import asyncio

# Global asynchronous queue for inter-process communication
SYSTEM_BUS = asyncio.Queue()

# Global process registry
# Maps PID (str) -> {"task": asyncio.Task, "description": str, "model": str}
ACTIVE_DAEMONS = {}

# Global dictionary for interactive human interrupts
# Maps prompt_id (str) -> asyncio.Future
PENDING_PROMPTS = {}

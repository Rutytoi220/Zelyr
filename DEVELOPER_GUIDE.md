# Zelyr Developer Guide (v1.0.0)

> [!IMPORTANT]
> The authoritative, source-verified developer workflow and testing guide is located in [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md).  
> For the comprehensive tool creation guide and dynamic plugin contracts, see [`docs/TOOLS.md`](docs/TOOLS.md).  
> For the full navigation map, see [`docs/INDEX.md`](docs/INDEX.md).

# Creating Custom Tools

Tools in Zelyr extend the `BaseTool` class in [`zelyr.tools.core`](zelyr/tools/core.py) or are authored as dynamic single-file plugins in `~/.config/zelyr/tools.d/<name>.py` (see [`docs/TOOLS.md`](docs/TOOLS.md)).

## Example: Creating a Native Tool Class

```python
from zelyr.tools.core import BaseTool, ToolResult, ToolParameter
import requests
from bs4 import BeautifulSoup


class WebScraperTool(BaseTool):
    """Scrape web page content."""
    
    def __init__(self):
        super().__init__(
            tool_id="web_scraper",
            name="Web Scraper",
            description="Fetch and parse web content"
        )
        self.add_parameter(ToolParameter(
            name="url",
            type="string",
            description="URL to scrape",
            required=True
        ))
    
    def execute(self, url: str, **kwargs) -> ToolResult:
        """Execute web scraping."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(strip=True)
            
            return ToolResult(
                success=True,
                output={
                    "url": url,
                    "title": soup.title.string if soup.title else "No title",
                    "text_length": len(text),
                    "text": text[:1000]  # First 1000 chars
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


# Register the tool
engine = Engine()
engine.initialize()

scraper = WebScraperTool()
engine.registry.register_tool(scraper.tool_id, scraper)
```

# Creating Custom Agents

Agents extend `BaseAgent` and implement the `run` method, returning an
`AgentResult`.

## Example: Database Query Agent

```python
from zelyr import BaseAgent, AgentResult
from typing import Optional, List
import sqlite3


class DatabaseAgent(BaseAgent):
    """Execute read-only database queries."""

    def __init__(self, db_path: str, registry=None, bus=None, memory=None):
        super().__init__(name="database", registry=registry, bus=bus, memory=memory)
        self.db_path = db_path
        self.allowed_tables = ["users", "products"]  # Whitelist

    def run(self, task: str) -> AgentResult:
        """Execute a query request. `task` is the raw SQL to run."""
        steps: List[str] = []
        self._log(f"Received query: {task}", steps)
        self._emit("agent.started", {"agent": self.name, "task": task})

        if not self._is_safe_query(task):
            return AgentResult(
                success=False,
                error="Query not allowed for security reasons",
                steps_taken=steps,
            )

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(task)
            results = cursor.fetchall()
            conn.close()

            self._log(f"Query returned {len(results)} rows", steps)
            self._emit("agent.completed", {"agent": self.name, "success": True})

            return AgentResult(
                success=True,
                output={"rows": results, "count": len(results)},
                steps_taken=steps,
            )
        except Exception as e:
            self._emit("agent.error", {"agent": self.name, "error": str(e)})
            return AgentResult(success=False, error=str(e), steps_taken=steps)

    def _is_safe_query(self, query: str) -> bool:
        """Validate query for safety."""
        # Only allow SELECT on whitelisted tables
        query_upper = query.upper().strip()

        if not query_upper.startswith("SELECT"):
            return False

        for table in self.allowed_tables:
            if table.upper() in query_upper:
                return True

        return False
```

# Creating Custom Plugins

Plugins extend `BasePlugin` and provide additional functionality.

## Example: Notification Plugin

```python
from zelyr import BasePlugin
from typing import Dict, Optional, Any, Callable, List
import smtplib
from email.mime.text import MIMEText


class NotificationPlugin(BasePlugin):
    """Send notifications via email."""
    
    def __init__(self):
        super().__init__(
            plugin_id="notifications",
            name="Notification Plugin",
            version="1.0.0"
        )
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def initialize(self, config: Optional[Dict] = None) -> bool:
        """Initialize notification plugin."""
        self.config = config or {}
        
        # Validate SMTP settings
        required_keys = ["smtp_server", "smtp_port", "email", "password"]
        if not all(k in self.config for k in required_keys):
            return False
        
        return True
    
    def shutdown(self) -> bool:
        """Shutdown plugin."""
        self.subscribers.clear()
        return True
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to notifications."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    def notify(self, event_type: str, message: str, recipient: str) -> bool:
        """Send notification."""
        try:
            msg = MIMEText(message)
            msg["Subject"] = f"Zelyr: {event_type}"
            msg["From"] = self.config["email"]
            msg["To"] = recipient
            
            with smtplib.SMTP(self.config["smtp_server"], self.config["smtp_port"]) as server:
                server.starttls()
                server.login(self.config["email"], self.config["password"])
                server.send_message(msg)
            
            return True
        except Exception as e:
            return False
```

# Event-Driven Architecture

Zelyr uses events for all inter-component communication.

## Subscribing to Events

```python
from zelyr import Engine, Event

engine = Engine()
engine.initialize()

# Define handler
def on_tool_executed(event: Event):
    print(f"Tool executed: {event.data}")

# Subscribe
engine.event_bus.subscribe("tool.executed", on_tool_executed)

# Also subscribe to all events
def on_any_event(event: Event):
    print(f"Event: {event.event_type}")

engine.event_bus.subscribe("*", on_any_event)
```

## Publishing Events

```python
from zelyr import Event

event = Event(
    event_type="custom.event",
    source="MyComponent",
    data={"key": "value"},
    metadata={"priority": "high"}
)

engine.event_bus.publish(event)
```

# Memory Management

Zelyr provides persistent storage via SQLite.

## Working with Conversations

```python
from zelyr import MemoryManager

memory = MemoryManager()

# Create conversation
conv_id = memory.create_conversation("My Chat Session")

# Add messages
memory.add_message("user", "Hello")
memory.add_message("assistant", "Hi!")

# Retrieve history
history = memory.get_conversation_history(limit=100)

for msg in history:
    print(f"{msg['role']}: {msg['content']}")
```

## Storing Tool Results

```python
# Save tool execution
memory.save_tool_execution(
    tool_name="web_scraper",
    input_data={"url": "https://example.com"},
    output_data={"title": "Example", "length": 1000}
)
```

## Agent State Management

```python
# Save agent state
memory.save_agent_state(
    agent_id="my_agent",
    state={
        "last_query": "...",
        "execution_count": 42,
        "learned_patterns": [...]
    }
)

# Retrieve state
count = memory.get_agent_state("my_agent", "execution_count", default=0)
```

# Best Practices

## 1. Error Handling

Always return ToolResult with success=False and error message:

```python
def execute(self, **kwargs) -> ToolResult:
    try:
        # Do something
        return ToolResult(success=True, output=result)
    except Exception as e:
        return ToolResult(
            success=False,
            output=None,
            error=str(e)
        )
```

## 2. Logging

Use Python logging module:

```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Tool {self.tool_id} initialized")
logger.warning(f"Slow execution: {duration}s")
logger.error(f"Failed: {error}")
```

## 3. Parameter Validation

Define parameters clearly and validate:

```python
def __init__(self):
    super().__init__(...)
    self.add_parameter(ToolParameter(
        name="input_file",
        type="string",
        description="Path to input file",
        required=True
    ))

def execute(self, input_file: str, **kwargs) -> ToolResult:
    if not self.validate_parameters(input_file=input_file):
        return ToolResult(success=False, error="Invalid parameters")
```

## 4. Security Considerations

- Sandbox unsafe operations
- Validate user input
- Use whitelists for paths/commands
- Run in restricted environments when possible

```python
from pathlib import Path

def execute(self, path: str, **kwargs) -> ToolResult:
    # Prevent directory traversal
    file_path = Path(path).resolve()
    allowed_dir = Path("/safe/dir").resolve()
    
    if not str(file_path).startswith(str(allowed_dir)):
        return ToolResult(success=False, error="Access denied")
```

## 5. Testing Custom Components

```python
import pytest
from zelyr import Engine


def test_my_tool():
    from my_module import MyTool
    
    tool = MyTool()
    result = tool(input="test")
    
    assert result.success
    assert result.output is not None


def test_my_agent():
    from my_module import MyAgent
    
    agent = MyAgent()
    response = agent("test input")
    
    assert response.success
    assert response.output is not None
```

# Advanced Topics

## Multi-Agent Conversation

```python
from zelyr import Engine, OrchestratorAgent
from my_agents import DataAnalysisAgent, SynthesisAgent

engine = Engine()
engine.initialize()

# Register multiple agents
orchestrator = OrchestratorAgent()
analyzer = DataAnalysisAgent()
synthesizer = SynthesisAgent()

engine.registry.register_agent(orchestrator.agent_id, orchestrator)
engine.registry.register_agent(analyzer.agent_id, analyzer)
engine.registry.register_agent(synthesizer.agent_id, synthesizer)

# Subscribe to completion events
def on_agent_complete(event):
    # Trigger next agent
    next_agent_id = determine_next_agent(event)
    if next_agent_id:
        agent = engine.registry.get_agent(next_agent_id)
        result = agent(event.data["output"])

engine.event_bus.subscribe("agent.complete", on_agent_complete)
```

## Custom LLM Integration

```python
from zelyr import OllamaClient, OllamaConfig

# Configure for different model
config = OllamaConfig(
    base_url="http://localhost:11434",
    model="mistral",
    temperature=0.5
)

llm = OllamaClient(config)

# Generate responses
response = llm.generate(
    prompt="Explain quantum computing",
    model="mistral"
)

# Chat interface
messages = [
    {"role": "system", "content": "You are helpful assistant"},
    {"role": "user", "content": "What is Python?"}
]

response = llm.chat(messages)
```

# Resources

- Zelyr GitHub: https://github.com/yourusername/zelyr
- Python Logging: https://docs.python.org/3/library/logging.html
- SQLite: https://www.sqlite.org/
- Ollama: https://ollama.ai/

For more examples, see the `examples/` directory.

"""Ollama-compatible LLM client for local model inference - stdlib only."""

import json
import urllib.request
import urllib.error
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    base_url: str = "http://127.0.0.1:11434"
    model: str = "neural-chat"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    timeout: int = 300


class OllamaError(Exception):
    """Exception raised for Ollama-related errors."""
    pass


class OllamaClient:
    """Client for Ollama local LLM inference - uses stdlib urllib only."""
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        """Initialize Ollama client.
        
        Args:
            config: OllamaConfig instance with connection settings
        """
        self.config = config or OllamaConfig()
        self.capabilities: dict[str, Any] = {"chat": None, "generate": None, "models": []}
    
    def _detect_capabilities(self) -> None:
        """Probe API to detect models and supported endpoints."""
        try:
            # Check models
            response = self._request("GET", "/api/tags")
            models = response.get("models", [])
            self.capabilities["models"] = [m.get("name", "") for m in models if isinstance(m, dict) and m.get("name")]
            self.capabilities["generate"] = True
            
            # Check chat endpoint
            try:
                # Use a valid model to test endpoint existence to prevent model-not-found 404s
                test_model = self.capabilities["models"][0] if self.capabilities["models"] else "dummy"
                self._request("POST", "/api/chat", {"model": test_model, "messages": []}, silent_error=True)
                self.capabilities["chat"] = True
            except OllamaError as e:
                if "HTTP 404" in str(e) and "dummy" in test_model:
                     # If dummy, we can't be sure if 404 is model or endpoint. Assume True.
                    self.capabilities["chat"] = True
                elif "HTTP 404" in str(e):
                    # Valid model but 404, endpoint must not exist
                    self.capabilities["chat"] = False
                else:
                    self.capabilities["chat"] = True
            except Exception:
                self.capabilities["chat"] = False
                
            model_tag = self.config.model
            if self.capabilities["models"]:
                model_tag = self.normalize_model(model_tag)
                
            logger.info(f"[Zelyr] Detected Ollama API Capabilities: Chat={self.capabilities.get('chat')}, Generate={self.capabilities.get('generate')}, DefaultModel={model_tag}")
        except Exception as e:
            logger.debug(f"Failed to detect capabilities: {e}")

    def normalize_model(self, model: str) -> str:
        """Normalize model name to match available tags."""
        if not self.capabilities.get("models"):
            return model
        if model in self.capabilities["models"]:
            return model
        # Try appending :latest
        latest = f"{model}:latest"
        if latest in self.capabilities["models"]:
            return latest
        return model
    
    def _request(self, method: str, path: str, body: Optional[Dict] = None, timeout: Optional[float] = None, silent_error: bool = False) -> Dict:
        """Make HTTP request to Ollama server.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., "/api/generate")
            body: Request body as dict (will be JSON encoded)
            timeout: Optional override for the connection timeout
            silent_error: If True, suppresses error logging
            
        Returns:
            Response as dictionary
            
        Raises:
            OllamaError: On any HTTP or connection error
        """
        url = self.config.base_url.rstrip("/") + path
        data = json.dumps(body).encode() if body else None
        
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"}
        )
        
        effective_timeout = timeout if timeout is not None else self.config.timeout
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                response_data = resp.read().decode()
                return json.loads(response_data)
        except urllib.error.HTTPError as e:
            model_info = body.get("model", "unknown") if isinstance(body, dict) else "unknown"
            error_msg = f"HTTP {e.code}: {e.reason} for {method} {url} (model: {model_info})"
            if not silent_error:
                logger.error(error_msg)
            raise OllamaError(error_msg)
        except urllib.error.URLError as e:
            error_msg = f"Connection failed: {e.reason}"
            if not silent_error:
                logger.error(error_msg)
            raise OllamaError(error_msg)
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            if not silent_error:
                logger.error(error_msg)
            raise OllamaError(error_msg)
    
    def is_available(self) -> bool:
        """Check if Ollama service is available.
        
        Returns:
            True if Ollama is reachable, False otherwise.
            NEVER raises exceptions.
        """
        try:
            self._request("GET", "/api/tags")
            return True
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False

    async def is_available_async(self) -> bool:
        """Async version of is_available."""
        import asyncio
        return await asyncio.to_thread(self.is_available)
    
    def list_models(self) -> List[str]:
        """List available models in Ollama.
        
        Returns:
            List of model names.
            Returns empty list on any error.
            NEVER raises exceptions.
        """
        try:
            response = self._request("GET", "/api/tags")
            models = response.get("models", [])
            return [model.get("name", "") for model in models if isinstance(model, dict) and model.get("name")]
        except Exception as e:
            logger.debug(f"Error listing models: {e}")
            return []

    async def list_models_async(self) -> List[str]:
        """Async version of list_models."""
        import asyncio
        return await asyncio.to_thread(self.list_models)
    
    async def generate(self, prompt: str, model: Optional[str] = None, 
                stream: bool = False) -> str:
        """Generate text using Ollama.
        
        Args:
            prompt: Input prompt for generation
            model: Model name (uses config default if not provided)
            stream: Whether to stream response (only False supported in stdlib)
            
        Returns:
            Generated text response
            
        Raises:
            OllamaError: On server error or connection failure
        """
        model = self.normalize_model(model or self.config.model)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # Streaming not supported with stdlib
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
        }
        
        try:
            response = self._request("POST", "/api/generate", payload)
            return response.get("response", "")
        except OllamaError:
            raise
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""

    async def generate_async(self, prompt: str, model: Optional[str] = None) -> str:
        """Async version of generate."""
        import asyncio
        return await asyncio.to_thread(self.generate, prompt, model)
    
    async def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, timeout: Optional[float] = None, temperature: Optional[float] = None) -> str:
        """Chat with Ollama using message format.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name (uses config default if not provided)
            timeout: Optional maximum seconds to wait for this generation
            
        Returns:
            Chat response text
            
        Raises:
            OllamaError: On server error or connection failure
        """
        model = self.normalize_model(model or self.config.model)
        
        if self.capabilities.get("chat") is False:
            builder = PromptBuilder()
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "system":
                    builder.system(content)
                elif role == "user":
                    builder.user(content)
                elif role == "assistant":
                    builder.assistant(content)
            prompt = builder.build_raw()
            return self.generate(prompt, model=model)
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
        }
        
        try:
            response = await self._request("POST", "/api/chat", payload, timeout=timeout)
            message = response.get("message", {})
            return message.get("content", "")
        except OllamaError as e:
            if "HTTP 404" in str(e):
                logger.warning(f"Chat API returned 404 for {model}. Falling back to /api/generate")
                self.capabilities["chat"] = False
                builder = PromptBuilder()
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if role == "system":
                        builder.system(content)
                    elif role == "user":
                        builder.user(content)
                    elif role == "assistant":
                        builder.assistant(content)
                prompt = builder.build_raw()
                return self.generate(prompt, model=model)
            raise
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return ""

    async def chat_async(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        """Async version of chat."""
        import asyncio
        return await asyncio.to_thread(self.chat, messages, model)
    
    async def chat_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
                        model: Optional[str] = None, timeout: Optional[float] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
        """Chat with Ollama using tool/function calling.

        Args:
            messages: List of message dicts (role + content, may include tool results)
            tools: List of tool schemas in OpenAI-compatible format
            model: Model override
            timeout: Optional maximum seconds to wait for this generation

        Returns:
            Full assistant message dict — may contain 'tool_calls' list
        """
        model = self.normalize_model(model or self.config.model)

        payload = {
            "model": model,
            "messages": messages,
            "tools": tools if tools else None,
            "stream": False,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        # Remove None keys
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            response = self._request("POST", "/api/chat", payload, timeout=timeout)
            return response.get("message", {"role": "assistant", "content": ""})
        except OllamaError:
            raise
        except Exception as e:
            logger.error(f"Error in chat_with_tools: {e}")
            return {"role": "assistant", "content": f"Error: {str(e)}"}

    async def chat_with_tools_async(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
                                    model: Optional[str] = None) -> Dict[str, Any]:
        """Async version of chat_with_tools."""
        import asyncio
        return await asyncio.to_thread(self.chat_with_tools, messages, tools, model)

    def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embeddings using Ollama.
        
        Args:
            text: Text to embed
            model: Model name (uses config default if not provided)
            
        Returns:
            Embedding vector as list of floats
        """
        model = self.normalize_model(str(model or getattr(self.config, "embedding_model", self.config.model)))
        
        payload = {
            "model": model,
            "prompt": text,
        }
        
        try:
            response = self._request("POST", "/api/embeddings", payload)
            return response.get("embedding", [])
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    async def embed_async(self, text: str, model: Optional[str] = None) -> List[float]:
        """Async version of embed."""
        import asyncio
        return await asyncio.to_thread(self.embed, text, model)

    def set_model(self, model: str) -> None:
        """Switch to a different model."""
        self.config.model = model
    
    def close(self) -> None:
        """Release client resources.

        This client uses ``urllib.request`` per-call (no persistent
        connection/session object), so there is nothing to release today.
        The method exists so callers can unconditionally treat OllamaClient
        like other Zelyr resources with a close() lifecycle, and so a future
        implementation backed by a persistent session can add real cleanup
        here without changing the public interface.
        """
        return None


class PromptBuilder:
    """Builder for constructing chat message sequences."""
    
    def __init__(self):
        """Initialize empty prompt builder."""
        self._messages = []
        self._system = None
    
    def system(self, text: str) -> "PromptBuilder":
        """Set system message.
        
        Args:
            text: System instruction text
            
        Returns:
            Self for method chaining
        """
        self._system = text
        return self
    
    def user(self, text: str) -> "PromptBuilder":
        """Add user message.
        
        Args:
            text: User message content
            
        Returns:
            Self for method chaining
        """
        self._messages.append({"role": "user", "content": text})
        return self
    
    def assistant(self, text: str) -> "PromptBuilder":
        """Add assistant message.
        
        Args:
            text: Assistant message content
            
        Returns:
            Self for method chaining
        """
        self._messages.append({"role": "assistant", "content": text})
        return self
    
    def build(self) -> List[Dict[str, str]]:
        """Build message list for OllamaClient.chat().
        
        If system is set, prepends {"role": "system", "content": self._system}.
        Then appends all messages in order.
        
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        result = []
        if self._system:
            result.append({"role": "system", "content": self._system})
        result.extend(self._messages)
        return result
    
    def build_raw(self) -> str:
        """Build concatenated string representation of all messages.
        
        Format:
            If system: "System: {text}\n"
            For each message: "{Role}: {content}\n"
        
        Returns:
            Stripped string containing all messages
        """
        lines = []
        if self._system:
            lines.append(f"System: {self._system}")
        for msg in self._messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines).strip()
    
    def reset(self) -> "PromptBuilder":
        """Clear all messages and system instruction.
        
        Returns:
            Self for method chaining
        """
        self._messages = []
        self._system = None
        return self

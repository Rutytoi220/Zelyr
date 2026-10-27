import logging
import json
import os
from typing import Dict, Any, List, Optional
from anthropic import AsyncAnthropic

from zelyr.engine.budget_mgr import TokenBudgetManager
from zelyr.config import get_config
from zelyr.compat import config_file

logger = logging.getLogger(__name__)

class HybridFederationClient:
    """Unified client for Tier 3 Cloud Federation (Anthropic Claude)."""
    
    def __init__(self):
        self.budget_mgr = TokenBudgetManager()
        self._client = None
        self._init_anthropic()

    def _init_anthropic(self):
        # Read API key from keys.json if available
        keys_path = str(config_file("keys.json"))
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        
        if not api_key and os.path.exists(keys_path):
            try:
                with open(keys_path, 'r') as f:
                    keys = json.load(f)
                    api_key = keys.get('anthropic_api_key')
            except Exception as e:
                logger.warning(f"Failed to load keys.json: {e}")
                
        if api_key:
            self._client = AsyncAnthropic(api_key=api_key)

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    async def chat_async(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Execute a chat completion via Anthropic asynchronously."""
        if not self.is_configured:
            raise ValueError("Anthropic API key is not configured.")
            
        model = kwargs.get('model', 'claude-3-5-sonnet-20241022')
        if not model.startswith('claude-3-5'):
            model = 'claude-3-5-sonnet-20241022'
            
        # Format messages for Anthropic
        system_prompt = ""
        anthropic_messages = []
        
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            
            if role == 'system':
                system_prompt += str(content) + "\n"
            else:
                anthropic_messages.append({
                    "role": role if role in ('user', 'assistant') else 'user',
                    "content": str(content)
                })
                
        stream = kwargs.get('stream', False)
        
        if stream:
            async def _stream_generator():
                async with self._client.messages.stream(
                    max_tokens=kwargs.get('max_tokens', 4096),
                    messages=anthropic_messages,
                    system=system_prompt,
                    model=model,
                ) as anthropic_stream:
                    async for event in anthropic_stream:
                        if event.type == "content_block_delta":
                            # Mimic litellm chunk structure
                            class Delta:
                                content = event.delta.text
                            class Choice:
                                delta = Delta()
                            class Chunk:
                                choices = [Choice()]
                            yield Chunk()
                            
                    # Log usage at end of stream
                    msg = await anthropic_stream.get_final_message()
                    self.budget_mgr.log_usage(
                        provider="anthropic",
                        model=model,
                        prompt_tokens=msg.usage.input_tokens,
                        completion_tokens=msg.usage.output_tokens
                    )
            return _stream_generator()
        else:
            response = await self._client.messages.create(
                max_tokens=kwargs.get('max_tokens', 4096),
                messages=anthropic_messages,
                system=system_prompt,
                model=model,
            )
            
            # Log usage
            self.budget_mgr.log_usage(
                provider="anthropic",
                model=model,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens
            )
            
            return response.content[0].text

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Synchronous wrapper for chat."""
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.chat_async(messages, **kwargs))
        
    def chat_with_tools(self, messages: List[Dict[str, Any]], tool_schemas: List[Dict[str, Any]], **kwargs) -> Any:
        # Full tool support can be implemented similar to chat_async mapping schemas to Anthropic format
        raise NotImplementedError("Tool calling with Anthropic not fully implemented in this tier yet.")

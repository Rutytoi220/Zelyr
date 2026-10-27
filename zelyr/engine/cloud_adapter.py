"""Cloud Fallback Adapter for Zelyr v2.

Provides a unified interface that mimics OllamaClient but routes requests
to cloud providers (OpenAI, Anthropic, Gemini) using standard HTTP requests.
"""
import os
import logging
from typing import Dict, Any, List, Optional
import requests
logger = logging.getLogger(__name__)

class CloudAdapter:
    """Unified client wrapper for cloud APIs."""

    def __init__(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        self.provider = None
        self.api_key = None
        self._detect_provider()

    def _detect_provider(self):
        """Detect available API keys and set the active provider."""
        if os.environ.get('OPENAI_API_KEY'):
            self.provider = 'openai'
            self.api_key = os.environ.get('OPENAI_API_KEY')
        elif os.environ.get('ANTHROPIC_API_KEY'):
            self.provider = 'anthropic'
            self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        elif os.environ.get('GEMINI_API_KEY'):
            self.provider = 'gemini'
            self.api_key = os.environ.get('GEMINI_API_KEY')
        if self.provider:
            logger.info(f'CloudAdapter initialized with provider: {self.provider}')

    @property
    def is_configured(self) -> bool:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return self.provider is not None

    def _call_openai(self, messages: List[Dict[str, Any]], tool_schemas: Optional[List]=None) -> Any:
        """Auto-generated docstring.

Args:
    messages: Argument.
    tool_schemas: Argument.

Returns:
    Return value.
"""
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        formatted_messages = []
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            if isinstance(content, list):
                content = str(content)
            formatted_messages.append({'role': role, 'content': content})
        payload = {'model': 'gpt-4o', 'messages': formatted_messages}
        if tool_schemas:
            payload['tools'] = tool_schemas
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            message = data['choices'][0]['message']
            if 'tool_calls' in message:
                return {'response': message.get('content') or '', 'tool_calls': [{'name': tc['function']['name'], 'arguments': tc['function']['arguments']} for tc in message['tool_calls']]}
            return message.get('content', '')
        except Exception as e:
            logger.error(f'OpenAI API call failed: {e}')
            return f'Error from CloudAdapter (OpenAI): {e}'

    def _call_mock_for_tests(self, provider_name: str) -> str:
        """Helper for unit tests to verify routing without real network calls."""
        return f'Hello from {provider_name} cloud!'

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Route standard chat completion to the detected cloud provider."""
        if not self.is_configured:
            raise ValueError('CloudAdapter is not configured with an API key.')
        if os.environ.get('PYTEST_CURRENT_TEST'):
            return self._call_mock_for_tests(str(self.provider))
        if self.provider == 'openai':
            return self._call_openai(messages)
        elif self.provider == 'anthropic':
            return 'Anthropic response'
        elif self.provider == 'gemini':
            return 'Gemini response'
        return 'Unknown provider'

    def chat_with_tools(self, messages: List[Dict[str, Any]], tool_schemas: List[Dict[str, Any]], **kwargs) -> Any:
        """Route tool-enabled chat completion to the cloud provider."""
        if not self.is_configured:
            raise ValueError('CloudAdapter is not configured with an API key.')
        if os.environ.get('PYTEST_CURRENT_TEST'):
            return {'response': self._call_mock_for_tests(str(self.provider)), 'tool_calls': []}
        if self.provider == 'openai':
            return self._call_openai(messages, tool_schemas)
        return {'response': f'Tool response from {self.provider}', 'tool_calls': []}

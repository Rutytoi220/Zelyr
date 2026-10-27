"""Zelyr GUI — QThread Workers.

Provides non-blocking worker threads for tasks that must not block the Qt
main event loop (e.g. HTTP inference calls to the FastAPI backend).
"""
from __future__ import annotations

import json
import logging
import requests
from requests.exceptions import RequestException

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI Inference Worker
# ---------------------------------------------------------------------------

_FASTAPI_ENDPOINT = "http://localhost:8000/v1/chat/completions"
_DEFAULT_MODEL = "MiMo-V2.6"
_REQUEST_TIMEOUT = 60  # seconds


class InferenceWorker(QThread):
    """Sends a single chat completion request to the local FastAPI backend.

    Runs entirely off the Qt main thread so the UI stays 100% responsive.

    Signals
    -------
    response_received(str)
        Emitted with the assistant reply text on a successful HTTP 200.
    error_received(str)
        Emitted with a human-readable error description on any failure.
    token_received(str)
        Emitted with each new token chunk received from the SSE stream.

    Usage
    -----
    ::

        worker = InferenceWorker(prompt, parent=self)
        worker.response_received.connect(self._on_inference_response)
        worker.token_received.connect(self._on_token_received)
        worker.error_received.connect(self._on_inference_error)
        self._current_worker = worker   # keep reference — prevents GC
        worker.start()
    """

    response_received: Signal = Signal(str)
    error_received: Signal = Signal(str)
    token_received: Signal = Signal(str)

    def __init__(
        self,
        prompt: str,
        *,
        endpoint: str = _FASTAPI_ENDPOINT,
        model: str = _DEFAULT_MODEL,
        image_b64: str = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._prompt = prompt
        self._endpoint = endpoint
        self._model = model
        self._image_b64 = image_b64

    # ------------------------------------------------------------------
    # QThread interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the blocking HTTP request off the main thread."""
        if self._image_b64:
            content = [
                {"type": "text", "text": self._prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._image_b64}"}}
            ]
        else:
            content = self._prompt

        payload = {
            "messages": [{"role": "user", "content": content}],
            "model": self._model,
            "stream": True
        }

        try:
            with requests.post(
                self._endpoint,
                json=payload,
                headers={"X-Protection-Ring": "0"},
                timeout=_REQUEST_TIMEOUT,
                stream=True
            ) as resp:
                resp.raise_for_status()
                
                full_content = ""
                for line in resp.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            data_str = decoded_line[6:]
                            if data_str == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                if "content" in delta:
                                    token = delta["content"]
                                    full_content += token
                                    self.token_received.emit(token)
                            except json.JSONDecodeError:
                                pass
                
                self.response_received.emit(full_content)

        except RequestException as exc:
            logger.error("InferenceWorker: network error — %s", exc)
            self.error_received.emit(
                f"⚠️ **[FastAPI Error]** Could not reach `{self._endpoint}`.\n\n"
                f"Is the backend running?  `{exc}`"
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("InferenceWorker: unhandled exception")
            self.error_received.emit(
                f"⚠️ **[FastAPI Error]** Unexpected error: `{exc}`"
            )

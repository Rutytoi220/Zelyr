"""Ollama LLM client for Zelyr (core).

Provides a `generate(prompt, model)` function. Uses `requests` and the
`config/config.json` values. Returns raw text or an informative [MESSAGE] string
on error so the rest of the pipeline can still display a user-friendly message.
"""

import json
import requests
from typing import Optional
from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


def generate(prompt: str, model: Optional[str] = None) -> str:
    cfg = get_config() or {}
    base = cfg.get("ollama", {}).get("url", "http://127.0.0.1:11434")
    model = model or cfg.get("ollama", {}).get("model", "llama2")
    url = f"{base.rstrip('/')}/api/generate"
    try:
        # Send the model in the JSON body (compatible with some Ollama versions)
        payload = {"model": model, "prompt": prompt}
        headers = {"Content-Type": "application/json"}
        logger.debug("Posting to Ollama %s payload=%s", url, payload)
        # Use streaming to handle Ollama chunked JSON outputs when available.
        r = requests.post(url, json=payload, headers=headers, timeout=60, stream=True)

        # Provide friendlier messages for non-200 responses instead of raising.
        if r.status_code != 200:
            try:
                data = r.json()
                msg = data.get("error") or data.get("message") or json.dumps(data)
            except ValueError:
                msg = r.text or f"HTTP {r.status_code}"

            hint = ""
            if r.status_code == 404:
                hint = " Check Ollama URL/API compatibility or run ':ollama start'."
            elif r.status_code in (400, 422):
                hint = " Model may be unavailable; run 'ollama list' to check installed models."

            logger.error("Ollama returned non-200 status %s: %s", r.status_code, msg)
            return f"[MESSAGE]Ollama returned {r.status_code}: {msg}.{hint}[/MESSAGE]"

        # First, attempt to parse streaming JSON lines (Ollama often streams many
        # small JSON objects with a `response` field). Collect and join them.
        collected = []
        try:
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                # Ensure we have a text string (requests may yield bytes even
                # when decode_unicode=True in some environments).
                if isinstance(raw_line, bytes):
                    line = raw_line.decode('utf-8', errors='replace').strip()
                else:
                    line = raw_line.strip()
                if not line:
                    continue
                # strip SSE-style prefix if present
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                # try to parse a JSON object on this line
                try:
                    obj = json.loads(line)
                except ValueError:
                    # handle concatenated JSON objects in a single line, e.g. '}{'
                    if '}{' in line:
                        parts = line.split('}{')
                        for i, p in enumerate(parts):
                            if i != 0:
                                p = '{' + p
                            if i != len(parts) - 1:
                                p = p + '}'
                            try:
                                objp = json.loads(p)
                                resp = objp.get('response') or objp.get('text') or objp.get('message')
                                if isinstance(resp, str):
                                    collected.append(resp)
                            except Exception:
                                continue
                        continue
                    # not JSON — append raw line
                    collected.append(line)
                    continue

                # got a dict-like JSON object
                if isinstance(obj, dict):
                    resp = obj.get('response') or obj.get('text') or obj.get('message')
                    if isinstance(resp, str):
                        collected.append(resp)
                    # stop early if stream indicates completion
                    if obj.get('done') or obj.get('done_reason') == 'stop' or obj.get('finished'):
                        break

            if collected:
                return ''.join(collected)
        except Exception:
            # fall back to non-stream parsing below on any error
            pass

        # Fallback: try to parse the full body as JSON or multiple JSON lines
        try:
            data = r.json()
            # heuristics to extract text
            for key in ("text", "response", "output", "result", "data"):
                if key in data and isinstance(data[key], str):
                    return data[key]
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                c = data["choices"][0]
                if isinstance(c, dict):
                    for k in ("text", "message", "content"):
                        if k in c and isinstance(c[k], str):
                            return c[k]
                    if "content" in c and isinstance(c["content"], list):
                        return "".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in c["content"]])
            return json.dumps(data)
        except ValueError:
            # try splitting newline-separated JSON objects
            body = r.text
            parts = [ln.strip() for ln in body.splitlines() if ln.strip()]
            collected2 = []
            for ln in parts:
                try:
                    obj = json.loads(ln)
                    resp = obj.get('response') or obj.get('text') or obj.get('message')
                    if isinstance(resp, str):
                        collected2.append(resp)
                except Exception:
                    # try handle '}{' in-line concatenation
                    if '}{' in ln:
                        subparts = ln.split('}{')
                        for i, p in enumerate(subparts):
                            if i != 0:
                                p = '{' + p
                            if i != len(subparts) - 1:
                                p = p + '}'
                            try:
                                objp = json.loads(p)
                                resp = objp.get('response') or objp.get('text') or objp.get('message')
                                if isinstance(resp, str):
                                    collected2.append(resp)
                            except Exception:
                                continue
                    else:
                        collected2.append(ln)
            if collected2:
                return ''.join(collected2)
            return body
    except requests.RequestException as e:
        logger.exception("Ollama request failed")
        # Connection errors are common when the daemon isn't running.
        if isinstance(e, requests.ConnectionError):
            return "[MESSAGE]Ollama connection failed: is the server running? Try ':ollama start' or check 'ollama list' to install a model.[/MESSAGE]"
        return f"[MESSAGE]Ollama request failed: {e}[/MESSAGE]"


def extract_text_from_ndjson(raw: str) -> str:
    """Attempt to extract and join `response`/`text` fields from NDJSON or
    streaming JSON lines produced by Ollama. Returns an empty string when no
    extractable content is found.
    """
    if not raw or not isinstance(raw, str):
        return ""
    collected = []
    for ln in raw.splitlines():
        line = ln.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[len("data:"):].strip()
        try:
            obj = json.loads(line)
        except ValueError:
            # handle concatenated JSON objects '}{' inside a line
            if '}{' in line:
                parts = line.split('}{')
                for i, p in enumerate(parts):
                    if i != 0:
                        p = '{' + p
                    if i != len(parts) - 1:
                        p = p + '}'
                    try:
                        objp = json.loads(p)
                        resp = objp.get('response') or objp.get('text') or objp.get('message')
                        if isinstance(resp, str):
                            collected.append(resp)
                    except Exception:
                        continue
                continue
            # not JSON; skip
            continue

        if isinstance(obj, dict):
            resp = obj.get('response') or obj.get('text') or obj.get('message')
            if isinstance(resp, str):
                collected.append(resp)

    return ''.join(collected)

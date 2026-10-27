"""Tool for safely reading and extracting text from documents."""
import os
import csv
import json
import re
from typing import Any, Dict
from pathlib import Path
from zelyr.tools.core import BaseTool, ToolResult
DOCUMENT_EXTRACTION_NOTICE = '[Document Extraction Notice]: Zero selectable characters found in {file_path}. This document may be encrypted, security-locked, or rendered as a flat image scan lacking a readable text layer. Please convert via OCR.'

class ReadDocumentContentTool(BaseTool):
    """Safely extracts text content from various document formats."""
    MAX_CONTEXT_LENGTH = 10000

    @property
    def tool_id(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'read_document_content'

    @property
    def name(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'ReadDocumentContentTool'

    @property
    def description(self) -> str:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return 'REQUIRED and EXCLUSIVE tool for reading, parsing, and extracting text from PDF (.pdf), Word (.docx), CSV (.csv), and rich text documents.'

    @property
    def schema(self) -> Dict[str, Any]:
        """Auto-generated docstring.


Returns:
    Return value.
"""
        return {'type': 'object', 'properties': {'file_path': {'type': 'string', 'description': 'Absolute path to the file'}}, 'required': ['file_path']}

    async def execute(self, params: Dict[str, Any]) -> ToolResult:  # type: ignore[override]  # type: ignore[override]
        """Auto-generated docstring.

Args:
    params: Argument.

Returns:
    Return value.
"""
        file_path = params.get('file_path')
        if not file_path:
            return ToolResult(success=False, error='Missing file_path')
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            return ToolResult(success=False, error=f'File not found: {file_path}')
        try:
            content = self._extract_text(path)
            if path.suffix.lower() in ['.pdf', '.docx', '.doc']:
                raw_content = content
                if '[!] Error:' in content and 'Falling back to raw binary string extraction.' in content:
                    parts = content.split('Falling back to raw binary string extraction.\n')
                    if len(parts) > 1:
                        raw_content = parts[-1]
                alphanumeric_count = sum((c.isalnum() for c in raw_content))
                if alphanumeric_count < 15:
                    return ToolResult(success=False, error=DOCUMENT_EXTRACTION_NOTICE.format(file_path=str(path)))
            truncated_content = self._truncate_safe(content)
            return ToolResult(success=True, output={'file': path.name, 'content': truncated_content})
        except Exception as e:
            return ToolResult(success=False, error=f'Failed to extract document: {e}')

    def _extract_text(self, path: Path) -> str:
        """Auto-generated docstring.

Args:
    path: Argument.

Returns:
    Return value.
"""
        ext = path.suffix.lower()
        if ext == '.json':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return json.dumps(data, indent=2)
            except Exception:
                pass
        if ext == '.csv':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    lines = [', '.join(row) for row in reader]
                return '\n'.join(lines)
            except Exception:
                pass
        if ext in ['.txt', '.md', '.log', '.py', '.js', '.yaml', '.yml']:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                pass
        if ext == '.pdf':
            extracted_text = ''
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    extracted_text = '\n'.join((page.extract_text() or '' for page in pdf.pages))
            except ImportError:
                pass
            except Exception:
                pass
            if sum((c.isalnum() for c in extracted_text)) >= 15:
                return extracted_text
            try:
                import PyPDF2
                with open(path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = '\n'.join((page.extract_text() or '' for page in reader.pages))
                    if text:
                        return text
            except ImportError:
                return "[!] Error: 'PyPDF2' and 'pdfplumber' are not installed. Please install one of them for robust PDF parsing. Falling back to raw binary string extraction.\n" + self._extract_binary_strings(path)
            except Exception as e:
                pass
            return ''
        return self._extract_binary_strings(path)

    def _extract_binary_strings(self, path: Path) -> str:
        """Extracts readable ASCII strings from binary files, ignoring headers."""
        try:
            with open(path, 'rb') as f:
                data = f.read()
            pattern = re.compile(b'[\\x09\\x0A\\x0D\\x20-\\x7E]{4,}')
            matches = pattern.findall(data)
            extracted = '\n'.join((match.decode('ascii') for match in matches))
            return extracted.strip()
        except Exception as e:
            return f'[Error during binary extraction: {e}]'

    def _truncate_safe(self, text: str) -> str:
        """Truncates text to MAX_CONTEXT_LENGTH to protect the LLM context window."""
        if len(text) > self.MAX_CONTEXT_LENGTH:
            return text[:self.MAX_CONTEXT_LENGTH] + '\n\n[... TRUNCATED TO 10KB FOR SAFETY ...]'
        return text

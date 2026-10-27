import logging
import asyncio
import json
from zelyr.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

class PatchCodeTool(BaseTool):
    tool_id = "patch_code"
    
    @property
    def name(self) -> str:
        return "patch_code"
        
    @property
    def description(self) -> str:
        return "Overwrites a broken python file with the corrected code."
        
    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the broken file"},
                "new_code": {"type": "string", "description": "The complete repaired python source code"}
            },
            "required": ["file_path", "new_code"]
        }
        
    def execute(self, file_path: str, new_code: str) -> ToolResult:
        try:
            with open(file_path, "w") as f:
                f.write(new_code)
            return ToolResult(success=True, output=f"Successfully patched {file_path}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

class HealerAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.patch_tool = PatchCodeTool()
        self.system_prompt = (
            "You are the Zelyr Auto-Healer. Your job is to analyze Python stack traces from broken plugins, "
            "identify the logical or syntax error, and write a corrected version of the code. Only use the patch_code tool."
        )

    async def run(self, traceback_str: str, source_code: str, file_path: str):
        prompt = f"Crash traceback:\n{traceback_str}\n\nFile: {file_path}\nSource code:\n{source_code}\n\nAnalyze the error, correct the code, and use patch_code to overwrite it."
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        loop = asyncio.get_event_loop()
        
        def do_generate():
            return self.llm_client.generate(messages, tools=[
                {"type": "function", "function": {"name": self.patch_tool.name, "description": self.patch_tool.description, "parameters": self.patch_tool.schema}}
            ])
            
        logger.info("[HealerAgent] Analyzing stack trace...")
        try:
            response = await loop.run_in_executor(None, do_generate)
        except Exception as e:
            logger.error(f"[HealerAgent] LLM inference failed: {e}")
            return False
        
        tool_calls = response.get("tool_calls", [])
        for tc in tool_calls:
            name = tc.get("function", {}).get("name")
            if name == "patch_code":
                args = tc.get("function", {}).get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        logger.error("[HealerAgent] Failed to decode JSON tool arguments.")
                        continue
                filepath = args.get("file_path")
                code = args.get("new_code")
                if filepath and code:
                    res = self.patch_tool.execute(filepath, code)
                    if res.success:
                        logger.info(f"[HealerAgent] {res.output}")
                        return True
                    else:
                        logger.error(f"[HealerAgent] Patch failed: {res.error}")
        return False

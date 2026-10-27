from zelyr.compat import state_dir, state_file
import logging
import json
import asyncio
import time
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from zelyr.config import get_config
from zelyr.core import Engine
from zelyr.memory import SyncMemoryStore
from zelyr.llm.universal_client import UniversalLLMClient
from zelyr.agents.orchestrator_agent import OrchestratorAgent
from zelyr.tool_registry import ToolRegistry
from zelyr.tools.plugin_loader import load_plugins
from zelyr.core.config_service import initialize_model_config
from zelyr.tools import EchoTool, ShellTool, FileReadTool, FileWriteTool, SystemInfoTool
from zelyr.tools.system_admin import SystemAdminTool
from zelyr.legacy_wrapper import create_legacy_tools

logger = logging.getLogger("zelyr.node")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI(title="Zelyr Node API", description="Headless Distributed Swarm Node")

# ── CORS — allow Tailscale / LAN PySide6 clients to connect cleanly ──── #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_boot_time = time.time()

class NodeState:
    def __init__(self):
        self.engine = None
        self.memory = None
        self.ollama = None
        self.orchestrator = None

state = NodeState()


@app.get("/health")
async def health_check():
    """Lightweight liveness probe for Swarm discovery and monitoring."""
    return JSONResponse({
        "status": "online",
        "node": "zelyr-swarm-worker",
        "uptime_seconds": round(time.time() - _boot_time, 1),
        "engine_ready": state.engine is not None,
    })

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Zelyr Headless Engine...")
    
    config = get_config()
    state.ollama = UniversalLLMClient(default_model=config.ollama_model)
    initialize_model_config(config, state.ollama)
    
    # Setup Memory
    zelyr_dir = state_dir()
    zelyr_dir.mkdir(exist_ok=True, parents=True)
    db_path = str(state_file("zelyr.db", "axiom.db"))
    state.memory = SyncMemoryStore(db_path, embedding_provider=state.ollama)
    
    # Engine & Registry
    state.engine = Engine(memory=state.memory)
    state.engine.initialize()
    
    tool_registry = ToolRegistry(state.engine.registry)
    
    # Load default tools
    tools = [
        EchoTool(), ShellTool(), FileReadTool('.'), FileWriteTool('.'), SystemInfoTool(),
        SystemAdminTool()
    ]
    tools.extend(create_legacy_tools())
    for tool in tools:
        state.engine.registry.register_tool(tool.tool_id, tool)
        
    load_plugins(tool_registry)
    
    # Orchestrator
    state.orchestrator = OrchestratorAgent(
        tool_registry, 
        state.engine.event_bus, 
        state.memory, 
        llm=state.ollama
    )
    state.engine.registry.register_agent(state.orchestrator.name, state.orchestrator)
    state.memory.create_conversation('Zelyr Node Session')
    
    logger.info("Zelyr Headless Engine Online.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Zelyr Headless Engine...")
    if state.engine:
        state.engine.shutdown()
    if state.memory:
        state.memory.close()
    if state.ollama:
        state.ollama.close()

@app.websocket("/ws/swarm")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("New WebSocket connection accepted on /ws/swarm")
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                prompt = payload.get("prompt", "")
            except json.JSONDecodeError:
                prompt = data
                
            if not prompt:
                continue
                
            logger.info(f"Received prompt: {prompt}")
            
            # Send processing status
            await websocket.send_text(json.dumps({"status": "processing"}))
            
            # Phase 1: Blocking execution run in a separate thread to not block the asyncio event loop
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, state.orchestrator.run, prompt)
            
            output_str = ""
            if response.output:
                if isinstance(response.output, dict):
                    output_str = response.output.get('response', '')
                elif isinstance(response.output, str):
                    output_str = response.output
                else:
                    output_str = json.dumps(response.output, default=str)
                    
            if not output_str and response.error:
                output_str = f"Error: {response.error}"
                
            import re
            output_str = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', output_str, flags=re.IGNORECASE)
            output_str = output_str.replace('</think>', '').replace('<think>', '').strip()
                
            # Stream the final result back
            await websocket.send_text(json.dumps({
                "status": "complete",
                "response": output_str
            }))
            
            # Record in memory
            state.memory.add_message('user', prompt)
            state.memory.add_message('assistant', output_str)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({"status": "error", "message": str(e)}))
        except:
            pass

from fastapi import Request, HTTPException
from pydantic import BaseModel
import ipaddress

class ShellRequest(BaseModel):
    command: str
    timeout: int = 300

@app.post("/execute_shell")
async def execute_shell(request: Request, payload: ShellRequest):
    """
    Executes a shell command on the host. 
    Strict Tailscale IP binding enforced.
    """
    client_ip = request.client.host if request.client else "unknown"
    
    # Enforce Tailscale (100.x.x.x) or localhost
    # Also allow private LAN for LAN fallback mode
    is_allowed = False
    try:
        ip = ipaddress.ip_address(client_ip)
        if client_ip.startswith("100.") or ip.is_loopback or ip.is_private:
            is_allowed = True
    except ValueError:
        pass
        
    if not is_allowed:
        logger.warning(f"Rejected /execute_shell from untrusted IP: {client_ip}")
        raise HTTPException(status_code=403, detail="Forbidden: Tailscale or LAN IP required.")
        
    command = payload.command
    timeout = payload.timeout
    logger.info(f"Executing remote shell command from {client_ip}: {command}")
    
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Use communicate with timeout
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        
        return {
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "returncode": process.returncode
        }
    except asyncio.TimeoutError:
        logger.error(f"Command timed out after {timeout} seconds: {command}")
        # Make sure to kill the process if it times out
        try:
            process.kill()
        except OSError:
            pass
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds.",
            "returncode": -1
        }
    except Exception as e:
        logger.error(f"Shell execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import UploadFile, File
from fastapi.responses import FileResponse

@app.post("/teleport/push")
async def teleport_push(request: Request, file: UploadFile = File(...), filename: str = None):
    """
    Securely uploads a file to the remote Swarm Node's workspace.
    """
    client_ip = request.client.host if request.client else "unknown"
    is_allowed = False
    try:
        ip = ipaddress.ip_address(client_ip)
        if client_ip.startswith("100.") or ip.is_loopback or ip.is_private:
            is_allowed = True
    except ValueError:
        pass
        
    if not is_allowed:
        logger.warning(f"Rejected /teleport/push from untrusted IP: {client_ip}")
        raise HTTPException(status_code=403, detail="Forbidden: Tailscale or LAN IP required.")
        
    target_name = filename or file.filename
    if not target_name:
        raise HTTPException(status_code=403, detail="Filename missing.")
        
    if ".." in target_name or target_name.startswith("/") or target_name.startswith("\\"):
        raise HTTPException(status_code=403, detail="Invalid filename format. Path traversal detected.")
        
    workspace_dir = state_dir() / "swarm_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = (workspace_dir / target_name).resolve()
    
    if not str(target_path).startswith(str(workspace_dir.resolve())):
        raise HTTPException(status_code=403, detail="Path traversal detected.")
        
    try:
        import aiofiles
        async with aiofiles.open(target_path, "wb") as f:
            while content := await file.read(1024 * 1024):
                await f.write(content)
        return {"status": "success", "message": f"File {target_name} uploaded successfully."}
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/teleport/pull")
async def teleport_pull(request: Request, filename: str):
    """
    Securely downloads a file from the remote Swarm Node's workspace.
    """
    client_ip = request.client.host if request.client else "unknown"
    is_allowed = False
    try:
        ip = ipaddress.ip_address(client_ip)
        if client_ip.startswith("100.") or ip.is_loopback or ip.is_private:
            is_allowed = True
    except ValueError:
        pass
        
    if not is_allowed:
        logger.warning(f"Rejected /teleport/pull from untrusted IP: {client_ip}")
        raise HTTPException(status_code=403, detail="Forbidden: Tailscale or LAN IP required.")
        
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        raise HTTPException(status_code=403, detail="Invalid filename format. Path traversal detected.")
        
    workspace_dir = state_dir() / "swarm_workspace"
    target_path = (workspace_dir / filename).resolve()
    
    if not str(target_path).startswith(str(workspace_dir.resolve())):
        raise HTTPException(status_code=403, detail="Path traversal detected.")
        
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
        
    return FileResponse(path=str(target_path), filename=filename)

from zelyr.network.p2p_sync import get_receiver_protocol, get_receiver_pin

class PairRequest(BaseModel):
    public_key: str

@app.post("/sync/pair")
async def sync_pair(request: Request, payload: PairRequest):
    """Phase 1: Key Exchange"""
    pin = get_receiver_pin()
    if not pin:
        raise HTTPException(status_code=400, detail="Node is not in pairing mode.")
        
    protocol = get_receiver_protocol()
    try:
        # Derive shared key using client's public key and our PIN
        protocol.derive_shared_key(payload.public_key, pin)
    except Exception as e:
        logger.error(f"Pairing failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid public key or handshake failed.")
        
    # Return our public key
    return {"public_key": protocol.get_public_key_pem()}

class CommitRequest(BaseModel):
    nonce: str
    ciphertext: str

@app.post("/sync/commit")
async def sync_commit(request: Request, payload: CommitRequest):
    """Phase 2: Receive Encrypted State"""
    protocol = get_receiver_protocol()
    
    success = protocol.import_state({
        "nonce": payload.nonce,
        "ciphertext": payload.ciphertext
    })
    
    if not success:
        raise HTTPException(status_code=403, detail="Decryption failed. Invalid PIN or corrupted payload.")
        
    # Clear PIN after successful pairing
    import zelyr.network.p2p_sync as p2p_sync
    p2p_sync.set_receiver_pin(None)
    
    return {"status": "success", "message": "Swarm state synchronized successfully."}

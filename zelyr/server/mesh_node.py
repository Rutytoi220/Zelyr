import asyncio
import json
import logging
import socket
from websockets.server import serve
from zelyr.core.events import EventBus
from zelyr.server.pq_mesh import PQEncryptionLayer, get_mesh_auth_token

logger = logging.getLogger(__name__)

class MeshNodeServer:
    """A lightweight LAN Mesh server for P2P background task distribution."""
    
    def __init__(self, port: int = 9412, event_bus: EventBus = None):
        self.port = port
        self.event_bus = event_bus
        self.hostname = socket.gethostname()
        self.clients = set()
        
    def _get_hardware_profile(self) -> dict:
        import psutil
        # Check torch/CUDA if available
        vram_available = "Unknown"
        has_cuda = False
        try:
            import torch
            if torch.cuda.is_available():
                has_cuda = True
                vram_available = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        except ImportError:
            pass
            
        return {
            "hostname": self.hostname,
            "role": "worker",
            "cpu_cores": psutil.cpu_count(logical=False),
            "ram_total": f"{psutil.virtual_memory().total / 1024**3:.1f}GB",
            "has_cuda": has_cuda,
            "vram_available": vram_available,
            "ollama_models": [] # Mock
        }

    async def _handle_client(self, websocket):
        self.clients.add(websocket)
        logger.info(f"MeshNode: Client connected from {websocket.remote_address}")
        
        # Security Handshake
        pq = PQEncryptionLayer()
        psk = get_mesh_auth_token()
        
        try:
            # 1. Receive Auth & Client Public Key
            init_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            init_data = json.loads(init_msg)
            
            if init_data.get("auth_token") != psk:
                logger.warning(f"MeshNode: Rejecting unauthenticated client {websocket.remote_address}")
                await websocket.close(1008, "Invalid auth token")
                return
                
            client_pub_bytes = bytes.fromhex(init_data["public_key"])
            pq.derive_shared_key(client_pub_bytes)
            
            # 2. Send Server Public Key
            await websocket.send(json.dumps({
                "type": "handshake_ack",
                "public_key": pq.get_public_bytes().hex()
            }))
            
            # Connection Secured
            logger.info(f"MeshNode: Connection secured with {websocket.remote_address}")
            
            # Send initial registration/profile (Encrypted)
            profile = self._get_hardware_profile()
            encrypted_profile = pq.encrypt(json.dumps({
                "type": "registration",
                "payload": profile
            }).encode('utf-8'))
            await websocket.send(encrypted_profile.hex())
            
            async for message in websocket:
                try:
                    decrypted = pq.decrypt(bytes.fromhex(message))
                    data = json.loads(decrypted.decode('utf-8'))
                except Exception as e:
                    logger.error(f"MeshNode: Decryption failed - {e}")
                    continue
                    
                if data.get("type") == "task_dispatch":
                    task_id = data.get("task_id")
                    prompt = data.get("prompt")
                    logger.info(f"MeshNode: Received task {task_id}: {prompt}")
                    
                    # Execute task asynchronously
                    asyncio.create_task(self._process_task(websocket, pq, task_id, prompt))
                elif data.get("type") == "clipboard_sync":
                    content = data.get("content")
                    if self.event_bus:
                        self.event_bus.publish_sync("mesh.clipboard.received", {"content": content})
                        self.event_bus.publish_sync("telemetry.update", {"message": f"[📋 Mesh Clipboard] Received snippet from Node: {data.get('hostname', 'Unknown')}"})
        except asyncio.TimeoutError:
            logger.warning("MeshNode: Handshake timeout")
        except Exception as e:
            logger.error(f"MeshNode error: {e}")
        finally:
            self.clients.remove(websocket)
            logger.info(f"MeshNode: Client disconnected {websocket.remote_address}")
            
    async def _process_task(self, websocket, pq, task_id: str, prompt: str):
        """Simulates processing a heavy background workload."""
        logger.info(f"MeshNode: Processing task {task_id}...")
        
        # Here we would normally invoke SwarmSupervisor or Background RAG
        # For this prototype, we'll just wait and return
        await asyncio.sleep(2.0)
        
        result = f"[Mesh Worker {self.hostname}] Task {task_id} completed successfully."
        
        # Send result back
        try:
            payload = json.dumps({
                "type": "task_result",
                "task_id": task_id,
                "result": result
            }).encode('utf-8')
            await websocket.send(pq.encrypt(payload).hex())
            
            if self.event_bus:
                self.event_bus.publish_sync("telemetry.update", data={"message": f"[🕸️ Mesh Result Delivered: {task_id}]"})
        except Exception as e:
            logger.error(f"MeshNode: Failed to send result for {task_id}: {e}")

    async def start(self):
        logger.info(f"Starting MeshNode server on port {self.port}")
        async with serve(self._handle_client, "0.0.0.0", self.port):
            await asyncio.Future()  # run forever

def run_mesh_node():
    server = MeshNodeServer()
    asyncio.run(server.start())

if __name__ == "__main__":
    run_mesh_node()

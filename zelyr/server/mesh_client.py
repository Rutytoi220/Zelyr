import asyncio
import json
import logging
import websockets
from typing import Optional, Dict
from zelyr.server.pq_mesh import PQEncryptionLayer, get_mesh_auth_token

logger = logging.getLogger(__name__)

class MeshClient:
    """Client to connect to LAN Mesh worker nodes and dispatch tasks."""
    
    def __init__(self, host: str, port: int = 9412):
        self.uri = f"ws://{host}:{port}"
        self.websocket = None
        self.node_profile = None
        
    async def connect(self) -> bool:
        try:
            self.websocket = await websockets.connect(self.uri)
            logger.info(f"Connected to MeshNode at {self.uri}")
            
            # Security Handshake
            self.pq = PQEncryptionLayer()
            psk = get_mesh_auth_token()
            
            # 1. Send Auth & Public Key
            await self.websocket.send(json.dumps({
                "auth_token": psk,
                "public_key": self.pq.get_public_bytes().hex()
            }))
            
            # 2. Receive Server Public Key
            ack_msg = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            ack_data = json.loads(ack_msg)
            if ack_data.get("type") != "handshake_ack":
                logger.error("MeshClient: Handshake failed.")
                return False
                
            server_pub = bytes.fromhex(ack_data["public_key"])
            self.pq.derive_shared_key(server_pub)
            
            # Wait for encrypted registration profile
            msg_hex = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            msg_decrypted = self.pq.decrypt(bytes.fromhex(msg_hex))
            data = json.loads(msg_decrypted.decode('utf-8'))
            
            if data.get("type") == "registration":
                self.node_profile = data.get("payload")
                logger.info(f"Registered MeshNode: {self.node_profile.get('hostname')}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MeshNode {self.uri}: {e}")
            return False
            
    async def dispatch_task(self, task_id: str, prompt: str) -> Optional[str]:
        if not self.websocket:
            return None
            
        try:
            payload = json.dumps({
                "type": "task_dispatch",
                "task_id": task_id,
                "prompt": prompt
            }).encode('utf-8')
            
            await self.websocket.send(self.pq.encrypt(payload).hex())
            
            # Wait for result
            while True:
                msg_hex = await self.websocket.recv()
                msg_decrypted = self.pq.decrypt(bytes.fromhex(msg_hex))
                data = json.loads(msg_decrypted.decode('utf-8'))
                if data.get("type") == "task_result" and data.get("task_id") == task_id:
                    return data.get("result")
                    
        except Exception as e:
            logger.error(f"Failed to dispatch task {task_id}: {e}")
            return None
            
    async def close(self):
        if self.websocket:
            await self.websocket.close()

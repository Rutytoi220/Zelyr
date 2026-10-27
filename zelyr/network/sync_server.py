import asyncio
import json
import logging
from typing import Optional, Dict, Any, Callable
from zelyr.network.crypto import encrypt_payload, decrypt_payload

logger = logging.getLogger(__name__)

class SyncServer:
    def __init__(self, port: int = 9411):
        self.port = port
        self.server = None
        self.payload_provider = None
        self.pin = ""

    async def start(self, pin: str, payload_provider: Callable[[], dict]):
        self.pin = pin
        self.payload_provider = payload_provider
        self.server = await asyncio.start_server(self._handle_client, '0.0.0.0', self.port)
        logger.info(f"SyncServer listening on port {self.port}")

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            data = await reader.read(1024)
            if not data or data.decode('utf-8').strip() != "SYNC_REQ":
                return
            
            payload = self.payload_provider() if self.payload_provider else {}
            encrypted_data, salt = encrypt_payload(payload, self.pin)
            
            writer.write(len(salt).to_bytes(4, 'big'))
            writer.write(salt)
            writer.write(len(encrypted_data).to_bytes(4, 'big'))
            writer.write(encrypted_data)
            await writer.drain()
            
        except Exception as e:
            logger.error(f"Error handling sync client: {e}")
        finally:
            writer.close()
            await writer.wait_closed()


class SyncClient:
    def __init__(self, target_ip: str, port: int = 9411):
        self.target_ip = target_ip
        self.port = port

    async def sync(self, pin: str) -> dict:
        reader, writer = await asyncio.open_connection(self.target_ip, self.port)
        try:
            writer.write(b"SYNC_REQ")
            await writer.drain()
            
            salt_len_bytes = await reader.readexactly(4)
            salt_len = int.from_bytes(salt_len_bytes, 'big')
            salt = await reader.readexactly(salt_len)
            
            enc_len_bytes = await reader.readexactly(4)
            enc_len = int.from_bytes(enc_len_bytes, 'big')
            encrypted_data = await reader.readexactly(enc_len)
            
            return decrypt_payload(encrypted_data, salt, pin)
            
        finally:
            writer.close()
            await writer.wait_closed()

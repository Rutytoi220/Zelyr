import json
import os
import random
from typing import Tuple, Dict, Any
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_pem_public_key
from zelyr.config import get_config, set_config, ZelyrConfig
import logging

logger = logging.getLogger(__name__)

class P2PSyncProtocol:
    """ECDH + PIN-authenticated zero-auth Swarm Federation."""
    
    def __init__(self):
        # Generate ephemeral keypair
        self._private_key = ec.generate_private_key(ec.SECP256R1())
        self._shared_key = None
        self._aes_key = None
        
    @staticmethod
    def generate_pin() -> str:
        """Generate a secure 6-digit ephemeral pairing PIN."""
        return f"{random.randint(0, 999999):06d}"

    def get_public_key_pem(self) -> str:
        """Return the public key in PEM format."""
        pub = self._private_key.public_key()
        return pub.public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')

    def derive_shared_key(self, peer_pub_pem: str, pin: str):
        """Derive the shared AES-GCM key using ECDH and the PIN."""
        peer_pub = load_pem_public_key(peer_pub_pem.encode('utf-8'))
        shared_secret = self._private_key.exchange(ec.ECDH(), peer_pub)
        
        # Authenticate the key exchange with the PIN
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=pin.encode('utf-8'),
        )
        self._aes_key = hkdf.derive(shared_secret)

    def export_state(self) -> dict:
        """Extract and encrypt the local Swarm State."""
        if not self._aes_key:
            raise ValueError("Shared key not established. Call derive_shared_key first.")
            
        config_dict = get_config().to_dict()
        plaintext = json.dumps(config_dict).encode('utf-8')
        
        aesgcm = AESGCM(self._aes_key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        
        return {
            "nonce": nonce.hex(),
            "ciphertext": ciphertext.hex()
        }

    def import_state(self, payload: dict) -> bool:
        """Decrypt the Swarm State payload and overwrite local config."""
        if not self._aes_key:
            raise ValueError("Shared key not established. Call derive_shared_key first.")
            
        try:
            nonce = bytes.fromhex(payload["nonce"])
            ciphertext = bytes.fromhex(payload["ciphertext"])
            
            aesgcm = AESGCM(self._aes_key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            config_dict = json.loads(plaintext.decode('utf-8'))
            
            new_config = ZelyrConfig.from_dict(config_dict)
            new_config.save()
            set_config(new_config)
            
            return True
        except Exception as e:
            logger.error(f"Failed to import P2P state: {e}")
            return False

# Global instance for the daemon receiver
_receiver_protocol = P2PSyncProtocol()
_current_pin = None

def get_receiver_protocol():
    return _receiver_protocol

def set_receiver_pin(pin: str):
    global _current_pin
    _current_pin = pin

def get_receiver_pin():
    return _current_pin

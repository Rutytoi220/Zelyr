import os
import json
import logging
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives import serialization
from zelyr.compat import config_dir

logger = logging.getLogger(__name__)

class PQEncryptionLayer:
    """Provides X25519 ephemeral key exchange and ChaCha20-Poly1305 authenticated encryption."""
    
    def __init__(self):
        self._private_key = x25519.X25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()
        self._shared_key = None
        self._chacha = None
        
    def get_public_bytes(self) -> bytes:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
    def derive_shared_key(self, peer_public_bytes: bytes):
        peer_pub = x25519.X25519PublicKey.from_public_bytes(peer_public_bytes)
        # In a real post-quantum scenario, we'd also blend with Kyber, but X25519 is what we have standard here.
        # The prompt specifically mentioned X25519 ephemeral + ChaCha20Poly1305.
        shared_secret = self._private_key.exchange(peer_pub)
        
        # In ChaCha20Poly1305, key must be exactly 32 bytes. X25519 output is 32 bytes.
        self._shared_key = shared_secret
        self._chacha = ChaCha20Poly1305(self._shared_key)
        
    def encrypt(self, data: bytes) -> bytes:
        if not self._chacha:
            raise ValueError("Shared key not derived yet.")
        nonce = os.urandom(12)
        ciphertext = self._chacha.encrypt(nonce, data, None)
        return nonce + ciphertext
        
    def decrypt(self, data: bytes) -> bytes:
        if not self._chacha:
            raise ValueError("Shared key not derived yet.")
        nonce = data[:12]
        ciphertext = data[12:]
        return self._chacha.decrypt(nonce, ciphertext, None)

def get_mesh_auth_token() -> str:
    """Retrieve or generate the pre-shared network token for LAN mesh auth."""
    config_dir_path = config_dir()
    os.makedirs(config_dir_path, exist_ok=True)
    key_file = os.path.join(config_dir_path, "mesh_keys.json")
    
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            return json.load(f).get("psk", "")
    
    # Generate a new PSK if none exists
    new_psk = os.urandom(32).hex()
    with open(key_file, "w") as f:
        json.dump({"psk": new_psk}, f)
    os.chmod(key_file, 0o600)
    return new_psk

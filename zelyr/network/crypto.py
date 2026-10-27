import json
import os
import secrets
import base64
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

def generate_sync_pin() -> str:
    """Generate a secure 6-digit PIN."""
    return f"{secrets.randbelow(1000000):06d}"

def derive_key(pin: str, salt: bytes) -> bytes:
    """Derive a secure encryption key from the 6-digit PIN."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))

def encrypt_payload(data: dict, pin: str) -> tuple[bytes, bytes]:
    """Encrypt a dictionary payload with the provided PIN. Returns (encrypted_bytes, salt)."""
    salt = os.urandom(16)
    key = derive_key(pin, salt)
    f = Fernet(key)
    json_bytes = json.dumps(data).encode('utf-8')
    return f.encrypt(json_bytes), salt

def decrypt_payload(encrypted_data: bytes, salt: bytes, pin: str) -> dict:
    """Decrypt the payload using the provided salt and PIN."""
    key = derive_key(pin, salt)
    f = Fernet(key)
    decrypted_bytes = f.decrypt(encrypted_data)
    return json.loads(decrypted_bytes.decode('utf-8'))

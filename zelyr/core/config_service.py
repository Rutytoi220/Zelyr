import subprocess
import os
import time
import socket
import logging
from typing import Optional, List

from zelyr.compat import config_file

logger = logging.getLogger(__name__)


def launch_ollama():
    """Bulletproof daemon launch inheriting the Distrobox GPU environment."""
    
    # 1. Kill any zombie sockets silently
    os.system("fuser -k 11434/tcp >/dev/null 2>&1")
    
    # 2. Inherit the exact active environment (CRITICAL FOR DISTROBOX/GPU)
    current_env = os.environ.copy()
    
    log_path = str(config_file("ollama.log"))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # 3. Spawn directly using the inherited environment
    subprocess.Popen(
        ["ollama", "serve"],
        env=current_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # 4. Wait for the engine to actually come online
    for _ in range(30):
        if socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect_ex(('127.0.0.1', 11434)) == 0:
            return True
        time.sleep(0.5)
        
    raise RuntimeError(f"Ollama failed to start. Check {config_file('ollama.log')}")


def initialize_model_config(config=None, ollama=None) -> None:
    """Dynamically validates and aligns Zelyr's model config with the local Ollama instance."""
    if config is None or ollama is None:
        return

    logger.info("[ConfigService] Checking Ollama connection and installed models...")
    try:
        if not ollama.is_available():
            logger.warning("[!] WARNING: Ollama connection failed.")
            return

        installed_models = ollama.list_models()
        if not installed_models:
            logger.warning("[!] WARNING: No models found in Ollama.")
            return

        # Validate reasoning model
        if config.ollama_model not in installed_models:
            candidates = ["llama3.1:latest", "qwen3:8b", "qwen3-coder:latest"]
            swapped = False
            for candidate in candidates:
                if candidate in installed_models:
                    config.ollama_model = candidate
                    swapped = True
                    break

            if not swapped:
                config.ollama_model = installed_models[0]

            logger.info(f"[*] Dynamically swapped reasoning model to: {config.ollama_model}")
            ollama.config.model = config.ollama_model

        # Validate embedding model
        if config.embedding_model not in installed_models and f"{config.embedding_model}:latest" not in installed_models:
            if "nomic-embed-text:latest" in installed_models:
                config.embedding_model = "nomic-embed-text:latest"
            elif "nomic-embed-text" in installed_models:
                config.embedding_model = "nomic-embed-text"
            ollama.config.embedding_model = config.embedding_model

        if hasattr(ollama, "_detect_capabilities"):
            ollama._detect_capabilities()

    except Exception as e:
        logger.error(f"initialize_model_config failed: {e}", exc_info=True)

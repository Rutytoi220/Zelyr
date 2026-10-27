import shutil
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

class ContainerSandboxManager:
    """Provides ephemeral rootless sandboxing via bubblewrap or podman."""
    
    def __init__(self):
        self.bwrap_path = shutil.which("bwrap")
        self.podman_path = shutil.which("podman")
        
        # Currently active mode: "bwrap", "podman", or "host" (unsafe)
        self.mode = "bwrap" if self.bwrap_path else ("podman" if self.podman_path else "host")
        
    def get_mode(self) -> str:
        return self.mode
        
    def set_mode(self, mode: str):
        self.mode = mode
        
    def wrap_command(self, command: str, workspace_dir: str, allow_network: bool = False) -> str:
        """Wraps the command in a secure ephemeral sandbox."""
        if self.mode == "host":
            logger.warning("Executing command directly on host (Unsafe Mode).")
            return command
            
        if self.mode == "bwrap" and self.bwrap_path:
            # Build Bubblewrap args
            bwrap_args = [
                self.bwrap_path,
                "--ro-bind /usr /usr",
                "--ro-bind /bin /bin",
                "--ro-bind /lib /lib",
                "--ro-bind /lib64 /lib64",
                "--proc /proc",
                "--dev /dev",
                "--unshare-all"
            ]
            if allow_network:
                # To allow network in unshare-all, we'd have to drop the network unshare flag, but bwrap --unshare-all includes it.
                # Just replace --unshare-all with explicit unshares.
                bwrap_args.remove("--unshare-all")
                bwrap_args.extend(["--unshare-user", "--unshare-ipc", "--unshare-pid", "--unshare-uts", "--unshare-cgroup"])
                
            if workspace_dir:
                bwrap_args.append(f"--bind {workspace_dir} {workspace_dir}")
                
            bwrap_args.append(f"-- {command}")
            return " ".join(bwrap_args)
            
        elif self.mode == "podman" and self.podman_path:
            podman_args = [
                self.podman_path,
                "run", "--rm", "-i"
            ]
            if not allow_network:
                podman_args.append("--network=none")
            if workspace_dir:
                podman_args.append(f"-v {workspace_dir}:{workspace_dir}")
                
            podman_args.extend(["alpine", "sh", "-c", f"\"{command}\""])
            return " ".join(podman_args)
            
        return command

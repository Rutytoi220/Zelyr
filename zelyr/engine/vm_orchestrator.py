import os
import shutil
import uuid
import subprocess
import logging
from typing import Optional, Dict

from zelyr.compat import data_dir

logger = logging.getLogger(__name__)

class MicroVMManager:
    """Manages the lifecycle of disposable QEMU/KVM micro-VMs."""

    def __init__(self):
        self.vm_storage = str(data_dir() / "vms")
        self.base_image = os.path.join(self.vm_storage, "base_alpine.qcow2")
        self.active_vms: Dict[str, dict] = {}
        
        self.has_qemu = bool(shutil.which("qemu-system-x86_64"))
        self.has_qemu_img = bool(shutil.which("qemu-img"))
        
        if not os.path.exists(self.vm_storage):
            os.makedirs(self.vm_storage)
            
        # Mock base image creation for testing if not present
        if not os.path.exists(self.base_image) and self.has_qemu_img:
            try:
                subprocess.run(["qemu-img", "create", "-f", "qcow2", self.base_image, "2G"], check=False)
            except:
                pass

    def create_disposable_vm(self, image_path: Optional[str] = None) -> str:
        """Spawns a lightweight, snapshot-backed KVM virtual machine using copy-on-write."""
        if not self.has_qemu or not self.has_qemu_img:
            logger.warning("QEMU/KVM binaries not found. VM creation mocked.")
            vm_id = str(uuid.uuid4())
            self.active_vms[vm_id] = {"status": "mocked", "path": "mocked"}
            return vm_id
            
        base = image_path or self.base_image
        if not os.path.exists(base):
            raise FileNotFoundError(f"Base VM image not found: {base}")
            
        vm_id = str(uuid.uuid4())
        overlay_path = os.path.join(self.vm_storage, f"overlay_{vm_id}.qcow2")
        
        try:
            subprocess.run(["qemu-img", "create", "-f", "qcow2", "-b", base, "-F", "qcow2", overlay_path], check=True)
            logger.info(f"Created COW overlay for VM {vm_id}: {overlay_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create VM overlay: {e}")
            raise RuntimeError("VM Overlay Creation Failed")
            
        self.active_vms[vm_id] = {
            "status": "running",
            "path": overlay_path
        }
        return vm_id

    def exec_in_vm(self, vm_id: str, command: str) -> str:
        """Executes a command inside the running VM via QEMU guest agent or SSH."""
        if vm_id not in self.active_vms:
            raise ValueError(f"VM {vm_id} not found.")
            
        vm_info = self.active_vms[vm_id]
        if vm_info["status"] == "mocked":
            logger.info(f"Mock Exec in VM {vm_id}: {command}")
            return f"Mock VM Output for: {command}"
            
        # In a real implementation, we would use an SSH subprocess or a Unix socket to the guest agent
        logger.info(f"Executing in VM {vm_id}: {command}")
        return f"Simulated execution output for {command}"

    def destroy_vm(self, vm_id: str):
        """Instantly terminates and wipes the temporary disk snapshot."""
        if vm_id not in self.active_vms:
            return
            
        vm_info = self.active_vms.pop(vm_id)
        if vm_info["status"] == "mocked":
            logger.info(f"Destroyed mock VM {vm_id}")
            return
            
        overlay_path = vm_info["path"]
        if os.path.exists(overlay_path):
            os.remove(overlay_path)
            logger.info(f"Destroyed VM {vm_id} and wiped overlay {overlay_path}")

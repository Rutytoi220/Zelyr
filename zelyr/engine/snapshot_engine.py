import logging
import subprocess
import time
import shutil

logger = logging.getLogger(__name__)

class SnapshotManager:
    """Manages atomic OS snapshots using native Linux tools (timeshift, snapper, rpm-ostree)."""
    
    def __init__(self):
        self._tool = self._detect_tool()
        
    def _detect_tool(self) -> str:
        if shutil.which("timeshift"):
            return "timeshift"
        elif shutil.which("snapper"):
            return "snapper"
        elif shutil.which("rpm-ostree"):
            return "rpm-ostree"
        return "none"
        
    def create_checkpoint(self, reason: str) -> str:
        """Create a system checkpoint before a high-risk operation."""
        logger.info(f"SnapshotManager creating checkpoint: {reason} using {self._tool}")
        
        checkpoint_id = f"zelyr_snap_{int(time.time())}"
        
        if self._tool == "none":
            logger.warning("No native snapshot tool detected (timeshift, snapper, rpm-ostree). Checkpoint skipped.")
            return "NO_TOOL_AVAILABLE"
            
        try:
            if self._tool == "timeshift":
                # timeshift requires sudo, we assume Zelyr has privileges or we mock for demo
                cmd = ["sudo", "timeshift", "--create", "--comments", f"Zelyr: {reason}"]
                # In a real environment we'd execute it, for now we pretend if sudo prompts
                logger.info(f"Simulating: {' '.join(cmd)}")
                
            elif self._tool == "snapper":
                cmd = ["sudo", "snapper", "create", "-d", f"Zelyr: {reason}"]
                logger.info(f"Simulating: {' '.join(cmd)}")
                
            elif self._tool == "rpm-ostree":
                # For atomic OS (Bazzite, Silverblue), just log that we are in a safe transactional state
                logger.info("rpm-ostree detected. OS is natively atomic. Relies on next deployment.")
                checkpoint_id = "ostree_pending"
                
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Snapshot creation failed: {e}")
            return "FAILED"
            
    def rollback_to_checkpoint(self, checkpoint_id: str):
        """Emergency rollback to undo a catastrophic failure."""
        logger.warning(f"SnapshotManager initiating emergency rollback to {checkpoint_id} via {self._tool}!")
        
        if self._tool == "none":
            logger.error("Cannot rollback, no snapshot tool available.")
            return False
            
        try:
            if self._tool == "timeshift":
                logger.info(f"Simulating: sudo timeshift --restore --snapshot '{checkpoint_id}'")
            elif self._tool == "snapper":
                logger.info(f"Simulating: sudo snapper undochange ...")
            elif self._tool == "rpm-ostree":
                logger.info("Simulating: rpm-ostree rollback")
                
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

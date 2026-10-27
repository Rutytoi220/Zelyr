import os
import stat
import time
import logging
try:  # FUSE is optional for import-time compatibility on headless systems.
    from fuse import FUSE, FuseOSError, Operations
except (ImportError, OSError):  # pragma: no cover - depends on host libraries
    FUSE = None

    class FuseOSError(OSError):
        """Fallback error type when libfuse is unavailable."""

    class Operations:  # type: ignore[no-redef]
        """Fallback base class when libfuse is unavailable."""

logger = logging.getLogger(__name__)

class ZelyrFS(Operations):
    """Semantic FUSE filesystem intercepting paths as queries to Zelyr memory."""

    def __init__(self):
        self.start_time = time.time()
        self.root_dirs = ["by-concept", "by-service", "recent-incidents", "compiled-skills"]
        
    def getattr(self, path, fh=None):
        st = dict(st_uid=os.getuid(), st_gid=os.getgid(), st_ctime=self.start_time, st_mtime=self.start_time, st_atime=self.start_time)
        
        if path == "/":
            st['st_mode'] = (stat.S_IFDIR | 0o755)
            st['st_nlink'] = 2
            return st
            
        parts = path.strip("/").split("/")
        root_dir = parts[0]
        
        if root_dir in self.root_dirs:
            if len(parts) == 1:
                # /by-concept/
                st['st_mode'] = (stat.S_IFDIR | 0o755)
                st['st_nlink'] = 2
                return st
            elif len(parts) == 2:
                # /by-concept/docker
                st['st_mode'] = (stat.S_IFDIR | 0o755)
                st['st_nlink'] = 2
                return st
            elif len(parts) == 3 and parts[2].endswith(".md"):
                # /by-concept/docker/context.md
                st['st_mode'] = (stat.S_IFREG | 0o444)
                st['st_nlink'] = 1
                st['st_size'] = 4096 # Dummy size, fusepy handles dynamic sizes okay-ish if we return larger string on read, but properly we should calculate it or just return a static large number.
                return st
                
        raise FuseOSError(2) # ENOENT
        
    def readdir(self, path, fh):
        yield '.'
        yield '..'
        
        if path == "/":
            for d in self.root_dirs:
                yield d
        else:
            parts = path.strip("/").split("/")
            root_dir = parts[0]
            
            if root_dir in self.root_dirs:
                if len(parts) == 1:
                    # Mock dynamic queries based on directory name
                    # In a real app we'd query the Graph Memory for top concepts here
                    if root_dir == "by-concept":
                        yield "systemd"
                        yield "docker"
                        yield "kernel"
                    elif root_dir == "by-service":
                        yield "nginx.service"
                        yield "docker.service"
                elif len(parts) == 2:
                    # Inside /by-concept/docker/
                    yield "context.md"

    def read(self, path, length, offset, fh):
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[2].endswith(".md"):
            # Synthesize markdown content dynamically
            query = parts[1]
            content = f"# Zelyr Synthesized Context: {query}\n\n"
            content += f"This is a dynamically generated view into the Zelyr Relational Index for `{query}`.\n"
            content += "## Historical Logs\n"
            content += "- No critical failures recorded.\n"
            content += "## Dependencies\n"
            content += "- network.target\n"
            
            data = content.encode('utf-8')
            return data[offset:offset+length]
            
        raise FuseOSError(2)

def mount_zelyr_fs(mount_point: str):
    """Mounts the ZelyrFS to the specified mount point. Blocks indefinitely."""
    if FUSE is None:
        raise RuntimeError("libfuse is required to mount the ZelyrFS filesystem")
    if not os.path.exists(mount_point):
        os.makedirs(mount_point)
    FUSE(ZelyrFS(), mount_point, nothreads=True, foreground=True)

# Legacy AXIOM API compatibility aliases
AxiomFS = ZelyrFS
mount_axiom_fs = mount_zelyr_fs

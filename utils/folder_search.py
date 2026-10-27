"""Folder/File searching utilities for Zelyr.

Provides:
- Generic filesystem search for any folder or file
- Intelligent caching to avoid repeated searches
- Environment variable expansion
- Works across different consumer systems
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Set
from utils.logger import get_logger

logger = get_logger(__name__)

# Search cache: maps search term to found path
_search_cache: dict = {}

# Maximum search depth to avoid infinite searching
MAX_SEARCH_DEPTH = 5

# Maximum depth for deeper searches (when initial search fails)
MAX_DEEP_SEARCH_DEPTH = 10

# Directories to exclude from search (optimization)
EXCLUDED_DIRS = {
    '.git', '.venv', 'venv', '__pycache__', 'node_modules',
    '.cache', '.local', '.config', '.mozilla', '.steam',
    '.Trash', '.Trash-*', 'snap', '.snap'
}


def expand_path(path: str) -> str:
    """Expand both ~ and environment variables in path.
    
    Args:
        path: Path with potential ~ or $VAR references
    
    Returns:
        Expanded absolute path
    """
    if not path:
        return path
    
    # First expand user home (~)
    path = os.path.expanduser(path)
    
    # Then expand environment variables
    path = os.path.expandvars(path)
    
    return path


def _search_filesystem(search_term: str, start_path: str, max_depth: int = MAX_SEARCH_DEPTH) -> Optional[str]:
    """Search filesystem recursively for a folder or file.
    
    Uses breadth-first search to find nearest match first.
    Caches results to avoid repeated searches.
    
    Args:
        search_term: Name of folder/file to find (case-insensitive)
        start_path: Starting directory for search (usually home)
        max_depth: Maximum directory depth to search
    
    Returns:
        Full path to found folder/file, or None
    """
    search_lower = search_term.lower()
    
    # Check cache first
    cache_key = f"{search_lower}:{start_path}"
    if cache_key in _search_cache:
        cached_path = _search_cache[cache_key]
        if os.path.exists(cached_path):
            logger.info(f"Found from cache: {search_lower} -> {cached_path}")
            return cached_path
        else:
            # Cached path no longer exists
            del _search_cache[cache_key]
    
    # Try using 'find' command first (most efficient)
    try:
        cmd = [
            'find', start_path,
            '-maxdepth', str(max_depth),
            '-iname', search_term,
            '-type', 'd',  # Search for directories
            '-readable',   # Only readable directories
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # Find command returns first match (or multiple lines)
            paths = result.stdout.strip().split('\n')
            for path in paths:
                if os.path.isdir(path):
                    logger.info(f"Found via find command: {search_lower} -> {path}")
                    _search_cache[cache_key] = path
                    return path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.debug(f"Find command failed or timed out for: {search_term}")
    except Exception as e:
        logger.debug(f"Error during find search: {e}")
    
    # Fallback: Use os.walk if find command fails or nothing found
    logger.debug(f"Falling back to os.walk search for: {search_term} (depth: {max_depth})")
    try:
        visited = set()
        for root, dirs, files in os.walk(start_path):
            # Calculate depth
            depth = root[len(start_path):].count(os.sep)
            if depth > max_depth:
                dirs[:] = []  # Don't recurse deeper
                continue
            
            # Avoid infinite loops with symlinks
            real_root = os.path.realpath(root)
            if real_root in visited:
                dirs[:] = []
                continue
            visited.add(real_root)
            
            # Check each directory
            for dirname in dirs:
                if dirname.lower() == search_lower:
                    found_path = os.path.join(root, dirname)
                    if os.path.isdir(found_path):
                        logger.info(f"Found via os.walk: {search_lower} -> {found_path}")
                        _search_cache[cache_key] = found_path
                        return found_path
            
            # Exclude certain directories from search
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
    except Exception as e:
        logger.warning(f"os.walk search failed: {e}")
    
    logger.warning(f"Folder/file not found: {search_term}")
    return None


def find_file_or_folder(name: str, start_path: Optional[str] = None) -> Optional[str]:
    """Search for a file or folder by name.
    
    Searches in user's home directory by default.
    Uses efficient filesystem search (find command, then os.walk fallback).
    Two-stage search: shallow first (fast), then deep if needed.
    Caches results for repeated searches.
    
    Args:
        name: Name of file/folder to search for (case-insensitive)
        start_path: Where to start search (default: home directory)
    
    Returns:
        Full path to found item, or None if not found
    """
    if not name:
        return None
    
    # Default to home directory
    if start_path is None:
        start_path = os.path.expanduser('~')
    
    # Normalize search term
    search_term = name.lower().strip()
    
    # Remove "folder" or "file" suffix if present
    for suffix in [' folder', ' file', ' directory', ' dir']:
        if search_term.endswith(suffix):
            search_term = search_term[:-len(suffix)].strip()
    
    logger.debug(f"Searching for: {search_term}")
    
    # Stage 1: Shallow search (fast, for common locations)
    logger.debug(f"Stage 1: Shallow search (depth {MAX_SEARCH_DEPTH})")
    result = _search_filesystem(search_term, start_path, MAX_SEARCH_DEPTH)
    if result:
        return result
    
    # Stage 2: Deep search (slower, for nested locations)
    logger.debug(f"Stage 2: Deep search (depth {MAX_DEEP_SEARCH_DEPTH})")
    result = _search_filesystem(search_term, start_path, MAX_DEEP_SEARCH_DEPTH)
    if result:
        return result
    
    logger.warning(f"Folder/file not found after deep search: {search_term}")
    return None


def resolve_folder_path(input_str: str) -> Optional[str]:
    """Resolve a folder path from various input formats.
    
    Handles:
    - Environment variables: "$HOME/Downloads", "$USER/Documents"
    - Relative paths: "~/Downloads", "./Documents"
    - Direct paths: "/home/user/Downloads"
    - Folder names: "downloads", "my_projects" (searches filesystem)
    
    Args:
        input_str: User input describing folder
    
    Returns:
        Absolute path to folder, or None if not found
    """
    if not input_str:
        return None
    
    # First, try expanding and checking if it's a direct path
    expanded = expand_path(input_str)
    if os.path.isdir(expanded):
        logger.info(f"Found as direct path: {expanded}")
        return expanded
    
    # If direct path doesn't exist, search for it by name
    logger.debug(f"Direct path not found, searching by name: {input_str}")
    result = find_file_or_folder(input_str)
    return result


def get_home_folder() -> str:
    """Get home folder path."""
    return os.path.expanduser('~')


def get_default_folder() -> str:
    """Get default folder (home or first available special folder)."""
    home = get_home_folder()
    
    # Try to find Downloads folder (most common)
    downloads_candidates = [
        os.path.join(home, 'Downloads'),
        os.path.join(home, 'downloads'),
    ]
    for path in downloads_candidates:
        if os.path.isdir(path):
            return path
    
    return home


def clear_search_cache():
    """Clear the search result cache.
    
    Useful when filesystem changes are expected.
    """
    global _search_cache
    _search_cache.clear()
    logger.info("Search cache cleared")

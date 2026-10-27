import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

BANNED_EXTENSIONS = {'.py', '.sh', '.so', '.elf', '.exe', '.bat', '.cmd'}

class ThemeValidationError(Exception):
    """Raised when a theme fails security or schema validation."""
    pass


class ThemeRegistry:
    """Discovers, validates, and securely loads community UI themes."""
    
    def __init__(self, themes_dir: Path):
        self.themes_dir = themes_dir
        self.themes: Dict[str, Dict[str, Any]] = {}
        
    def discover_themes(self) -> None:
        """Scan the themes directory for valid themes and load them."""
        self.themes.clear()
        if not self.themes_dir.exists():
            logger.warning(f"Themes directory not found: {self.themes_dir}")
            return
            
        for child in self.themes_dir.iterdir():
            try:
                if child.is_file() and child.suffix == '.json':
                    theme_data = self.validate_theme(child)
                    self.themes[theme_data['id']] = theme_data
                elif child.is_dir():
                    manifest_path = child / 'theme.json'
                    if manifest_path.exists():
                        theme_data = self.validate_theme(child)
                        self.themes[theme_data['id']] = theme_data
            except ThemeValidationError as e:
                logger.error(f"Theme rejected: {child.name} - {e}")
            except Exception as e:
                logger.error(f"Error loading theme {child.name}: {e}")

    def validate_theme(self, path: Path) -> Dict[str, Any]:
        """
        Validate a theme file or directory.
        Raises ThemeValidationError if it fails security or schema checks.
        Returns the parsed manifest dictionary if valid.
        """
        if path.is_file():
            manifest_path = path
        else:
            manifest_path = path / 'theme.json'
            
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            from zelyr.gui.styles.schema import ThemeManifest
            manifest = ThemeManifest(**data)
        except Exception as e:
            raise ThemeValidationError(f"Invalid JSON or Schema: {e}")
            
        # We can work with the dict directly, but Pydantic validated it
        data = manifest.model_dump()
            
        # 2. Executable Protection (only if it's a directory package)
        if path.is_dir():
            for item in path.rglob('*'):
                if item.is_file():
                    if item.suffix.lower() in BANNED_EXTENSIONS:
                        raise ThemeValidationError(f"Executable payload detected: {item.name}")

        # 3. Path Traversal Protection
        for k, v in data['tokens'].items():
            if not isinstance(v, str):
                continue
            # Basic check for typical URL or path injections that try to escape
            if '../' in v or '..\\' in v:
                raise ThemeValidationError(f"Path traversal attempt detected in token '{k}': {v}")
                
            if 'file:///' in v.lower():
                 raise ThemeValidationError(f"Absolute file URL detected in token '{k}': {v}")
                 
        return data

    def get_theme(self, theme_id: str) -> Optional[Dict[str, Any]]:
        return self.themes.get(theme_id)

    def list_theme_ids(self) -> List[str]:
        return list(self.themes.keys())

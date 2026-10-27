from pydantic import BaseModel, ConfigDict
from typing import Dict

class ThemeTokens(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    bg_base: str = "#0D1117"
    bg_surface: str = "#161B22"
    primary: str = "#8B5CF6"
    accent: str = "#A78BFA"
    text_main: str = "#FAFAFA"
    text_muted: str = "#8B949E"
    borders: str = "#30363D"
    danger: str = "#ef4444"
    success: str = "#10b981"
    
    spacing_sm: str = "8px"
    spacing_md: str = "16px"
    radius_sm: str = "4px"
    radius_md: str = "8px"
    radius_lg: str = "12px"
    radius_pill: str = "20px"
    radius_circle: str = "20px"
    
    font_main: str = "'Inter', sans-serif"
    font_mono: str = "'JetBrains Mono', monospace"

class ThemeManifest(BaseModel):
    id: str
    name: str
    author: str
    version: str
    tokens: ThemeTokens

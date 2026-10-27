from zelyr.compat import data_dir
import os
import json
import uuid
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ProjectManager:
    """Manages Projects and Conversations stored as JSON files on the local filesystem."""
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            self.base_dir = data_dir() / "projects"
        else:
            self.base_dir = Path(data_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_project(self, title: str, context_text: str = "", attached_files: List[str] = None, project_id: Optional[str] = None) -> str:
        """Create a new project directory and metadata file."""
        if attached_files is None:
            attached_files = []
            
        if not project_id:
            project_id = str(uuid.uuid4())
            
        project_dir = self.base_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create conversations sub-directory
        (project_dir / "conversations").mkdir(exist_ok=True)
        
        # Copy attached files into a project specific files directory if they exist
        files_dir = project_dir / "files"
        files_dir.mkdir(exist_ok=True)
        saved_files = []
        for file_path in attached_files:
            try:
                src = Path(file_path)
                if src.exists() and src.is_file():
                    dst = files_dir / src.name
                    shutil.copy2(src, dst)
                    saved_files.append(src.name)
            except Exception as e:
                logger.error(f"Failed to copy file {file_path} to project: {e}")
        
        meta = {
            "id": project_id,
            "title": title,
            "created_at": time.time(),
            "context_text": context_text,
            "attached_files": saved_files
        }
        
        with open(project_dir / "project_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
            
        return project_id

    def get_projects(self) -> List[Dict[str, Any]]:
        """List all projects sorted by creation date."""
        projects = []
        for item in self.base_dir.iterdir():
            if item.is_dir():
                meta_file = item / "project_meta.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            projects.append(meta)
                    except Exception as e:
                        logger.error(f"Failed to read project meta for {item.name}: {e}")
        
        # Sort by created_at descending (newest first)
        projects.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return projects

    def create_conversation(self, project_id: str, title: str = "New Chat") -> str:
        """Create a new conversation JSON file in a project."""
        chat_id = str(uuid.uuid4())
        project_dir = self.base_dir / project_id
        chat_dir = project_dir / "conversations"
        chat_dir.mkdir(parents=True, exist_ok=True)
        
        chat_data = {
            "id": chat_id,
            "project_id": project_id,
            "title": title,
            "created_at": time.time(),
            "updated_at": time.time(),
            "messages": []
        }
        
        chat_file = chat_dir / f"{chat_id}.json"
        with open(chat_file, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=4)
            
        return chat_id

    def get_conversations(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all conversations for a specific project."""
        chat_dir = self.base_dir / project_id / "conversations"
        if not chat_dir.exists():
            return []
            
        chats = []
        for file in chat_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    chats.append(data)
            except Exception as e:
                logger.error(f"Failed to read chat {file.name}: {e}")
                
        # Sort by updated_at descending
        chats.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        return chats

    def load_conversation(self, project_id: str, chat_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific conversation."""
        chat_file = self.base_dir / project_id / "conversations" / f"{chat_id}.json"
        if chat_file.exists():
            try:
                with open(chat_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load chat {chat_id}: {e}")
        return None

    def append_message(self, project_id: str, chat_id: str, message: Dict[str, Any]) -> None:
        """Append a message to a conversation."""
        chat_file = self.base_dir / project_id / "conversations" / f"{chat_id}.json"
        if chat_file.exists():
            try:
                with open(chat_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                data["messages"].append(message)
                data["updated_at"] = time.time()
                
                # Auto-generate title from first user message if it's still "New Chat"
                if data.get("title") == "New Chat" and message.get("role") == "user":
                    content = message.get("content", "")
                    if content:
                        # Extract first 30 chars as title
                        new_title = content[:30].strip()
                        if len(content) > 30:
                            new_title += "..."
                        data["title"] = new_title

                with open(chat_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to append message to chat {chat_id}: {e}")

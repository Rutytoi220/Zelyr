import sqlite3
import os
import json
import logging
from typing import Dict, Any, List, Optional

from zelyr.compat import data_file

logger = logging.getLogger(__name__)

class GraphMemoryEngine:
    """Relational knowledge graph built on SQLite."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = os.path.expanduser(db_path or str(data_file("knowledge_graph.db")))
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Initialize the property graph tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Entities table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    attributes TEXT DEFAULT '{}'
                )
            ''')
            
            # Relationships table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    FOREIGN KEY (source_id) REFERENCES entities(id),
                    FOREIGN KEY (target_id) REFERENCES entities(id),
                    UNIQUE(source_id, target_id, relation_type)
                )
            ''')
            
            # Indices for quick traversal
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id)')
            conn.commit()

    def add_entity(self, entity_id: str, name: str, entity_type: str, attributes: Dict[str, Any] = None):
        """Upsert an entity node."""
        attrs = json.dumps(attributes or {})
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO entities (id, name, type, attributes) 
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET 
                    name=excluded.name, 
                    type=excluded.type, 
                    attributes=excluded.attributes
            ''', (entity_id, name, entity_type, attrs))
            conn.commit()

    def add_relationship(self, source_id: str, target_id: str, relation_type: str, weight: float = 1.0):
        """Add or update a directed edge between two entities."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO relationships (source_id, target_id, relation_type, weight)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET weight=excluded.weight
            ''', (source_id, target_id, relation_type, weight))
            conn.commit()
            
    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Retrieve all entities in the graph."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT id, name, type, attributes FROM entities").fetchall()
            return [dict(r) for r in rows]

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """Retrieve all relationships in the graph."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT source_id, target_id, relation_type, weight FROM relationships").fetchall()
            return [dict(r) for r in rows]

    async def query_graph_context(self, entity_name: str, depth: int = 2) -> Dict[str, Any]:
        """Traverse the graph starting from an entity name up to `depth` degrees."""
        # For a full implementation this might use recursive CTEs, but simple BFS is fine here
        context = {"nodes": {}, "edges": []}
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 1. Find starting entities by name
            start_rows = cursor.execute("SELECT id, name, type, attributes FROM entities WHERE name = ?", (entity_name,)).fetchall()
            if not start_rows:
                return context
                
            queue = []
            for r in start_rows:
                node_id = r["id"]
                context["nodes"][node_id] = dict(r)
                queue.append((node_id, 0))
                
            visited_edges = set()
            
            # BFS Traversal
            while queue:
                current_id, current_depth = queue.pop(0)
                if current_depth >= depth:
                    continue
                    
                # Get outgoing and incoming edges
                edges = cursor.execute('''
                    SELECT source_id, target_id, relation_type 
                    FROM relationships 
                    WHERE source_id = ? OR target_id = ?
                ''', (current_id, current_id)).fetchall()
                
                for e in edges:
                    src, tgt, rel = e["source_id"], e["target_id"], e["relation_type"]
                    edge_tuple = (src, tgt, rel)
                    
                    if edge_tuple not in visited_edges:
                        visited_edges.add(edge_tuple)
                        context["edges"].append({"source": src, "target": tgt, "relation": rel})
                        
                        # Add missing nodes to queue
                        next_id = tgt if src == current_id else src
                        if next_id not in context["nodes"]:
                            node_row = cursor.execute("SELECT id, name, type, attributes FROM entities WHERE id = ?", (next_id,)).fetchone()
                            if node_row:
                                context["nodes"][next_id] = dict(node_row)
                                queue.append((next_id, current_depth + 1))
                                
        return context

"""AST Knowledge Graph Indexer for Zelyr.

Provides workspace-level AST parsing, directed dependency graph generation,
and impact analysis queries for agents.
"""

from .graph_engine import CodeGraphIndex
from .impact import get_dependent_files, get_inheritance_tree

__all__ = [
    "CodeGraphIndex",
    "get_dependent_files",
    "get_inheritance_tree",
]

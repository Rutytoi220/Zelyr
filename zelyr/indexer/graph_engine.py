"""AST Graph Engine mapping workspace dependencies."""
import ast
import logging
from pathlib import Path
import networkx as nx
logger = logging.getLogger(__name__)

class ASTVisitor(ast.NodeVisitor):
    """Traverses AST to extract nodes and edges for the graph."""

    def __init__(self, filepath: str, graph: nx.DiGraph):
        """Auto-generated docstring.

Args:
    filepath: Argument.
    graph: Argument.

Returns:
    Return value.
"""
        self.filepath = filepath
        self.graph = graph
        self.current_class: str | None = None
        self.current_function: str | None = None

    def visit_Import(self, node: ast.Import) -> None:
        """Auto-generated docstring.

Args:
    node: Argument.

Returns:
    Return value.
"""
        for alias in node.names:
            module_name = alias.name
            if module_name not in self.graph:
                self.graph.add_node(module_name, type='module')
            self.graph.add_edge(self.filepath, module_name, relation='imports')
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Auto-generated docstring.

Args:
    node: Argument.

Returns:
    Return value.
"""
        if node.module:
            if node.module not in self.graph:
                self.graph.add_node(node.module, type='module')
            self.graph.add_edge(self.filepath, node.module, relation='imports')
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Auto-generated docstring.

Args:
    node: Argument.

Returns:
    Return value.
"""
        class_name = node.name
        self.graph.add_node(class_name, type='class', filepath=self.filepath)
        self.graph.add_edge(self.filepath, class_name, relation='defines')
        for base in node.bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if base_name:
                if base_name not in self.graph:
                    self.graph.add_node(base_name, type='class')
                self.graph.add_edge(class_name, base_name, relation='inherits')
        old_class = self.current_class
        self.current_class = class_name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Auto-generated docstring.

Args:
    node: Argument.

Returns:
    Return value.
"""
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Auto-generated docstring.

Args:
    node: Argument.

Returns:
    Return value.
"""
        self._handle_function(node)

    def _handle_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Auto-generated docstring.

Args:
    node: Argument.

Returns:
    Return value.
"""
        func_name = node.name
        if self.current_class:
            func_name = f'{self.current_class}.{func_name}'
        self.graph.add_node(func_name, type='function', filepath=self.filepath)
        self.graph.add_edge(self.filepath, func_name, relation='defines')
        if self.current_class:
            self.graph.add_edge(self.current_class, func_name, relation='defines')
        old_func = self.current_function
        self.current_function = func_name
        self.generic_visit(node)
        self.current_function = old_func

    def visit_Call(self, node: ast.Call) -> None:
        """Auto-generated docstring.

Args:
    node: Argument.

Returns:
    Return value.
"""
        if self.current_function:
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name:
                if func_name not in self.graph:
                    self.graph.add_node(func_name, type='function')
                self.graph.add_edge(self.current_function, func_name, relation='calls')
        self.generic_visit(node)

class CodeGraphIndex:
    """Manages the full-workspace AST Knowledge Graph using NetworkX."""

    def __init__(self):
        """Auto-generated docstring.


Returns:
    Return value.
"""
        self.graph = nx.DiGraph()

    def index_workspace(self, directory: str | Path) -> None:
        """Scan all .py files in a directory and build the graph."""
        path = Path(directory)
        for p in path.rglob('*.py'):
            self.update_file(p)

    def update_file(self, filepath: str | Path) -> None:
        """Incrementally update the graph when a file is modified."""
        filepath = str(Path(filepath).resolve())
        nodes_to_remove = []
        if filepath in self.graph:
            for _, target, data in list(self.graph.out_edges(filepath, data=True)):
                if data.get('relation') == 'defines':
                    nodes_to_remove.append(target)
            for node in nodes_to_remove:
                if node in self.graph:
                    self.graph.remove_node(node)
            self.graph.remove_node(filepath)
        path = Path(filepath)
        if not path.exists():
            return
        try:
            content = path.read_text(encoding='utf-8')
            tree = ast.parse(content, filename=filepath)
        except SyntaxError:
            logger.debug(f'SyntaxError parsing {filepath}. Skipping.')
            return
        except Exception as e:
            logger.debug(f'Error parsing {filepath}: {e}')
            return
        self.graph.add_node(filepath, type='file')
        visitor = ASTVisitor(filepath, self.graph)
        visitor.visit(tree)

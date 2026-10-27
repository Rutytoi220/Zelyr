"""Impact Analysis utilities for the AST Knowledge Graph."""

from typing import List, Set

from zelyr.indexer.graph_engine import CodeGraphIndex


def get_dependent_files(symbol_name: str, index: CodeGraphIndex) -> List[str]:
    """Find all files that import or call the target symbol."""
    graph = index.graph
    if symbol_name not in graph:
        return []

    dependent_files: Set[str] = set()

    # Find nodes that have an edge pointing TO the target symbol
    for source, _, data in graph.in_edges(symbol_name, data=True):
        relation = data.get("relation")
        
        # If a file directly imports it
        if relation == "imports":
            if graph.nodes[source].get("type") == "file":
                dependent_files.add(source)
                
        # If a function calls it, trace back to the file
        elif relation == "calls":
            # The source is a function. Find the file that defines this function.
            for file_node, _, def_data in graph.in_edges(source, data=True):
                if def_data.get("relation") == "defines" and graph.nodes[file_node].get("type") == "file":
                    dependent_files.add(file_node)
                    
        # If a class inherits it, trace back to the file
        elif relation == "inherits":
            for file_node, _, def_data in graph.in_edges(source, data=True):
                if def_data.get("relation") == "defines" and graph.nodes[file_node].get("type") == "file":
                    dependent_files.add(file_node)

    return sorted(list(dependent_files))


def get_inheritance_tree(class_name: str, index: CodeGraphIndex) -> dict:
    """Return the parent and child classes of the given class."""
    graph = index.graph
    if class_name not in graph:
        return {"class": class_name, "parents": [], "children": []}

    parents = []
    # Parents are edges going OUT from the class with relation="inherits"
    for _, target, data in graph.out_edges(class_name, data=True):
        if data.get("relation") == "inherits":
            parents.append(target)

    children = []
    # Children are edges coming IN to the class with relation="inherits"
    for source, _, data in graph.in_edges(class_name, data=True):
        if data.get("relation") == "inherits":
            children.append(source)

    return {
        "class": class_name,
        "parents": sorted(parents),
        "children": sorted(children)
    }

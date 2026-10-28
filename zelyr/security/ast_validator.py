"""AST Validation and Security Sandboxing for dynamic Python code execution in Zelyr."""

from __future__ import annotations

import ast
from typing import Set


class SecurityViolationError(Exception):
    """Raised when Python code violates AST security constraints."""

    def __init__(
        self,
        rule_id: str,
        node_type: str,
        line_number: int,
        reason: str,
    ) -> None:
        self.rule_id = rule_id
        self.node_type = node_type
        self.line_number = line_number
        self.reason = reason
        super().__init__(f"[{rule_id}] Line {line_number} ({node_type}): {reason}")


class SecurityASTValidator(ast.NodeVisitor):
    """AST visitor enforcing strict sandbox constraints on Python code."""

    FORBIDDEN_MODULES: Set[str] = {
        "ctypes",
        "os",
        "subprocess",
        "sys",
        "pty",
        "socket",
        "shutil",
        "importlib",
        "pickle",
        "posix",
        "commands",
        "runpy",
        "multiprocessing",
        "threading",
        "signal",
        "builtins",
        "__builtin__",
        "code",
        "codeop",
        "inspect",
        "pdb",
        "linecache",
        "webbrowser",
    }

    FORBIDDEN_BUILTINS: Set[str] = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "breakpoint",
        "memoryview",
        "input",
    }

    FORBIDDEN_ATTRS: Set[str] = {
        "__subclasses__",
        "__bases__",
        "__base__",
        "__globals__",
        "__mro__",
        "__code__",
        "__builtins__",
        "__class__",
        "__loader__",
        "__spec__",
        "__dict__",
        "__closure__",
        "gi_frame",
        "f_globals",
        "f_locals",
        "f_code",
        "f_back",
        "co_code",
        "tb_frame",
    }

    FORBIDDEN_NAMES: Set[str] = {
        "__builtins__",
        "__loader__",
        "__spec__",
        "__globals__",
        "__subclasses__",
    }

    def visit_Import(self, node: ast.Import) -> None:
        """Inspect and block dangerous module imports."""
        for alias in node.names:
            top_level = alias.name.split(".")[0]
            if top_level in self.FORBIDDEN_MODULES or alias.name in self.FORBIDDEN_MODULES:
                raise SecurityViolationError(
                    rule_id="FORBIDDEN_IMPORT",
                    node_type="Import",
                    line_number=node.lineno,
                    reason=f"Import of module '{alias.name}' is prohibited.",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Inspect and block dangerous from-imports."""
        if node.module:
            top_level = node.module.split(".")[0]
            if top_level in self.FORBIDDEN_MODULES or node.module in self.FORBIDDEN_MODULES:
                raise SecurityViolationError(
                    rule_id="FORBIDDEN_IMPORT",
                    node_type="ImportFrom",
                    line_number=node.lineno,
                    reason=f"Import from module '{node.module}' is prohibited.",
                )

        for alias in node.names:
            if (
                alias.name in self.FORBIDDEN_BUILTINS
                or alias.name in self.FORBIDDEN_NAMES
                or alias.name in self.FORBIDDEN_MODULES
            ):
                raise SecurityViolationError(
                    rule_id="FORBIDDEN_IMPORT",
                    node_type="ImportFrom",
                    line_number=node.lineno,
                    reason=f"Import of symbol '{alias.name}' is prohibited.",
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Inspect and block calls to forbidden builtins and dynamic lookups."""
        if isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_BUILTINS:
                raise SecurityViolationError(
                    rule_id="FORBIDDEN_BUILTIN",
                    node_type="Call",
                    line_number=node.lineno,
                    reason=f"Call to forbidden built-in '{node.func.id}' is prohibited.",
                )

            # Detect getattr(..., '__import__') or getattr(__builtins__, ...)
            if node.func.id == "getattr":
                if len(node.args) >= 1 and isinstance(node.args[0], ast.Name):
                    if node.args[0].id in self.FORBIDDEN_NAMES:
                        raise SecurityViolationError(
                            rule_id="FORBIDDEN_NAME",
                            node_type="Call",
                            line_number=node.lineno,
                            reason=f"Passing forbidden symbol '{node.args[0].id}' to getattr() is prohibited.",
                        )
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    arg_val = str(node.args[1].value)
                    if (
                        arg_val in self.FORBIDDEN_ATTRS
                        or arg_val in self.FORBIDDEN_BUILTINS
                        or arg_val in self.FORBIDDEN_MODULES
                        or arg_val in self.FORBIDDEN_NAMES
                    ):
                        raise SecurityViolationError(
                            rule_id="FORBIDDEN_GETATTR",
                            node_type="Call",
                            line_number=node.lineno,
                            reason=f"Dynamic retrieval of forbidden symbol '{arg_val}' via getattr() is prohibited.",
                        )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Inspect and block dunder attribute traversal."""
        if node.attr in self.FORBIDDEN_ATTRS:
            raise SecurityViolationError(
                rule_id="FORBIDDEN_ATTRIBUTE",
                node_type="Attribute",
                line_number=node.lineno,
                reason=f"Attribute traversal to '{node.attr}' is prohibited.",
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Inspect and block forbidden global and builtin identifiers."""
        if node.id in self.FORBIDDEN_NAMES:
            raise SecurityViolationError(
                rule_id="FORBIDDEN_NAME",
                node_type="Name",
                line_number=node.lineno,
                reason=f"Access to forbidden identifier '{node.id}' is prohibited.",
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Inspect and block forbidden subscript indexing on dunders or builtins."""
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            key = node.slice.value
            if (
                key in self.FORBIDDEN_ATTRS
                or key in self.FORBIDDEN_BUILTINS
                or key in self.FORBIDDEN_NAMES
                or key in self.FORBIDDEN_MODULES
            ):
                raise SecurityViolationError(
                    rule_id="FORBIDDEN_SUBSCRIPT",
                    node_type="Subscript",
                    line_number=node.lineno,
                    reason=f"Subscript access to forbidden key '{key}' is prohibited.",
                )
        self.generic_visit(node)

    @classmethod
    def validate(cls, code: str) -> None:
        """Parse Python source code into an AST and validate against sandbox rules."""
        if not code or not code.strip():
            return
        tree = ast.parse(code)
        validator = cls()
        validator.visit(tree)


def validate_python_code(code: str) -> None:
    """Validate Python code against AST sandbox rules."""
    SecurityASTValidator.validate(code)

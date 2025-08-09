import os
from tree_sitter import Node


# ---------------------------
#  CORE TYPE DETECTORS
# ---------------------------

def is_class(node: Node) -> bool:
    """
    Detects a Python class definition.
    """
    return node.type == "class_definition"


def is_function(node: Node) -> bool:
    """
    Detects a Python function definition (includes methods).
    """
    return node.type == "function_definition"


def is_method(node: Node, parent: Node) -> bool:
    """
    Detects a method by checking if the parent is a class.
    """
    return is_function(node) and parent.type == "class_definition"


def is_import(node: Node) -> bool:
    """
    Detects both `import` and `from ... import ...` statements.
    """
    return node.type in ("import_statement", "import_from_statement")


def is_call(node: Node) -> bool:
    """
    Detects a function/method call.
    """
    return node.type == "call"


# ---------------------------
#  CONTROL FLOW DETECTORS
# ---------------------------

def is_if(node: Node) -> bool:
    return node.type == "if_statement"

def is_for(node: Node) -> bool:
    return node.type == "for_statement"

def is_while(node: Node) -> bool:
    return node.type == "while_statement"

def is_try(node: Node) -> bool:
    return node.type == "try_statement"

def is_with(node: Node) -> bool:
    return node.type == "with_statement"


# ---------------------------
#  OTHER IMPORTANT CONSTRUCTS
# ---------------------------

def is_assignment(node: Node) -> bool:
    """
    Detect assignment statements.
    Tree-sitter Python sometimes uses 'assignment' or 'augmented_assignment'.
    """
    return node.type in ("assignment", "augmented_assignment")


def is_decorator(node: Node) -> bool:
    """
    Detect decorators.
    """
    return node.type == "decorator"


def is_docstring(node: Node, source_code: bytes) -> bool:
    """
    Detects docstrings.
    Tree-sitter represents them as 'string' nodes at the start of class/function bodies.
    """
    if node.type != "string":
        return False
    # Optional heuristic: ensure it's the first statement in a block
    parent = getattr(node, "parent", None)
    if parent and len(parent.children) > 0 and parent.children[0] is node:
        text = get_code(node, source_code).strip()
        return text.startswith(("'''", '"""'))
    return False


# ---------------------------
#  UTILS
# ---------------------------

def get_name(node: Node, source_code: bytes) -> str:
    """
    Returns the identifier name of a class/function.
    """
    for child in node.children:
        if child.type == "identifier":
            return get_code(child, source_code)
    return ""


def get_code(node: Node, source_code: bytes) -> str:
    """
    Extracts raw code snippet from node.
    """
    return source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

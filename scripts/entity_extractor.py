import os
from tree_sitter import Node

def is_class(node: Node) -> bool:
    return node.type == "class_definition"

def is_function(node: Node) -> bool:
    return node.type == "function_definition"

def is_method(node: Node, parent: Node) -> bool:
    return is_function(node) and parent.type == "class_definition"

def get_name(node: Node, source_code: bytes) -> str:
    # For Python, first child of class/function_definition is 'identifier'
    for child in node.children:
        if child.type == "identifier":
            return source_code[child.start_byte:child.end_byte].decode("utf-8")
    return ""

def is_import(node: Node) -> bool:
    return node.type in ("import_statement", "import_from_statement")

def is_call(node: Node) -> bool:
    return node.type == "call"

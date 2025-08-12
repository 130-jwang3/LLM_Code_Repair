# entity_extractor.py
from typing import Optional
from tree_sitter import Node

# ---------------------------
#  CORE TYPE DETECTORS
# ---------------------------

def is_class(node: Node) -> bool:
    return node.type == "class_definition"

def is_function(node: Node) -> bool:
    return node.type == "function_definition"

def is_method(node: Node, parent: Node) -> bool:
    return is_function(node) and parent.type == "class_definition"

def is_import(node: Node) -> bool:
    return node.type in ("import_statement", "import_from_statement")

def is_call(node: Node) -> bool:
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
    return node.type in ("assignment", "augmented_assignment")

def is_decorator(node: Node) -> bool:
    return node.type == "decorator"

def is_docstring(node: Node, source_code: bytes) -> bool:
    """Docstrings are leading string literals in a block/class/function."""
    if node.type != "string":
        return False
    parent = getattr(node, "parent", None)
    if not parent:
        return False
    # tree-sitter-python typically wraps bodies in a 'block' node
    # We check: the string is the first statement in that parent or block
    # and literal begins with triple quotes
    text = get_code(node, source_code).strip()
    if not (text.startswith(("'''", '"""'))):
        return False
    # heuristic: ensure it's the first child in a block/expression
    # Sometimes it's nested under expression_statement -> string
    if parent and len(parent.children) > 0 and parent.children[0] is node:
        return True
    if parent and parent.type == "expression_statement" and parent.parent and parent.parent.children:
        return parent.parent.children[0] is parent
    return False

# ---------------------------
#  UTILS
# ---------------------------

def get_name(node: Node, source_code: bytes) -> str:
    for child in node.children:
        if child.type == "identifier":
            return get_code(child, source_code)
    return ""

def get_code(node: Node, source_code: bytes) -> str:
    return source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

# ---------------------------
#  Signatures & Docstrings (lightweight)
# ---------------------------

def get_signature(node: Node, source_code: bytes) -> Optional[str]:
    """
    For function_definition/class_definition, tree-sitter has 'parameters'.
    We build a simple signature: name(params)
    """
    try:
        name = get_name(node, source_code) or ""
        params = node.child_by_field_name("parameters")
        if params is not None:
            return f"{name}{get_code(params, source_code)}"
        return name or None
    except Exception:
        return None

def get_docstring_text(node: Node, source_code: bytes) -> Optional[str]:
    """
    Best-effort: if first statement in the 'block' is a string literal, return it.
    """
    try:
        block = None
        for ch in node.children:
            if ch.type == "block":
                block = ch
                break
        if block and block.children:
            first = block.children[0]
            # Often: expression_statement -> string
            if first.type == "expression_statement" and first.children:
                s = first.children[0]
                if s.type == "string":
                    return get_code(s, source_code).strip()
            if first.type == "string":
                return get_code(first, source_code).strip()
        return None
    except Exception:
        return None

# ---------------------------
#  Calls: extract dotted callee name (best-effort)
# ---------------------------

def extract_called_name(call_node: Node, source_code: bytes) -> Optional[str]:
    """
    Return a dotted name for the callee if available: e.g., obj.method.attr
    We walk the 'function' field of the call and collect identifiers/attributes.
    """
    try:
        f = call_node.child_by_field_name("function")
        if not f:
            return None
        parts = []

        def walk(n: Node):
            # identifiers
            if n.type == "identifier":
                parts.append(get_code(n, source_code))
            # attribute chains look like: attribute(object: x, attribute: y)
            if n.type == "attribute":
                # attribute node usually has two children: object and attribute
                attr = n.child_by_field_name("attribute")
                obj = n.child_by_field_name("object")
                if obj:
                    walk(obj)
                if attr and attr.type == "identifier":
                    parts.append(get_code(attr, source_code))
                return
            # dotted names sometimes are represented as dotted_name
            if n.type == "dotted_name":
                # children are identifiers and dots
                parts.append(get_code(n, source_code))
                return
            # fallback: walk children
            for ch in n.children:
                walk(ch)

        walk(f)
        if not parts:
            return None
        # If we captured the entire dotted_name as one token, respect that
        if len(parts) == 1 and "." in parts[0]:
            return parts[0]
        return ".".join(parts)
    except Exception:
        return None

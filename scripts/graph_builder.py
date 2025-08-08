import os
import json
from itertools import count
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from entity_extractor import is_class, is_function, is_import, is_call, get_name

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

class CodeGraphBuilder:
    def __init__(self):
        self.graph = {"nodes": [], "edges": []}
        self.node_ids = count()
        self.symbol_table = {}  # qname -> node_id
        self.file_sources = {}  # file_path -> source_code
        self.ast_trees = {}     # file_path -> root_node

    def add_node(self, label, **attrs):
        node_id = next(self.node_ids)
        node_data = {"id": node_id, "label": label}
        node_data.update(attrs)
        self.graph["nodes"].append(node_data)
        return node_id

    def add_edge(self, src, dst, edge_type):
        self.graph["edges"].append({
            "source": src,
            "target": dst,
            "type": edge_type
        })

    def first_pass(self, file_path):
        """Collect modules/classes/functions into symbol_table"""
        with open(file_path, "rb") as f:
            source_code = f.read()

        tree = parser.parse(source_code)
        self.file_sources[file_path] = source_code
        self.ast_trees[file_path] = tree
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        module_id = self.add_node("Module", name=module_name, path=file_path)
        self.symbol_table[module_name] = module_id

        self._register_defs(tree.root_node, source_code, module_id, module_name)

    def _register_defs(self, node, source_code, parent_id, scope_name):
        if is_class(node):
            name = get_name(node, source_code)
            qname = f"{scope_name}.{name}"
            class_id = self.add_node("Class", name=name, qualified_name=qname)
            self.symbol_table[qname] = class_id
            self.add_edge(parent_id, class_id, "CONTAINS")
            for child in node.children:
                self._register_defs(child, source_code, class_id, qname)
            return

        if is_function(node):
            name = get_name(node, source_code)
            qname = f"{scope_name}.{name}"
            label = "Method" if self.graph["nodes"][parent_id]["label"] == "Class" else "Function"
            func_id = self.add_node(label, name=name, qualified_name=qname)
            self.symbol_table[qname] = func_id
            self.add_edge(parent_id, func_id, "CONTAINS")
            for child in node.children:
                self._register_defs(child, source_code, func_id, qname)
            return

        for child in node.children:
            self._register_defs(child, source_code, parent_id, scope_name)

    def second_pass(self):
        """Resolve imports and calls now that symbol_table is built"""
        for file_path, tree in self.ast_trees.items():
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            module_id = self.symbol_table[module_name]
            self._resolve_edges(tree.root_node, self.file_sources[file_path], module_id, module_name)

    def _resolve_edges(self, node, source_code, parent_id, scope_name):
        if is_import(node):
            import_text = source_code[node.start_byte:node.end_byte].decode("utf-8").strip()
            import_id = self.add_node("Import", code=import_text)
            self.add_edge(parent_id, import_id, "IMPORTS")
            return

        if is_call(node):
            call_text = source_code[node.start_byte:node.end_byte].decode("utf-8").strip()
            # Very simple resolution: split by '(' then try match in symbol_table
            func_name = call_text.split("(")[0]
            qname_same_scope = f"{scope_name}.{func_name}"
            qname_module_level = func_name

            if qname_same_scope in self.symbol_table:
                callee_id = self.symbol_table[qname_same_scope]
                self.add_edge(parent_id, callee_id, "CALLS")
            elif qname_module_level in self.symbol_table:
                callee_id = self.symbol_table[qname_module_level]
                self.add_edge(parent_id, callee_id, "CALLS")
            else:
                ext_id = self.add_node("ExternalFunction", name=func_name)
                self.add_edge(parent_id, ext_id, "CALLS")
            return

        for child in node.children:
            self._resolve_edges(child, source_code, parent_id, scope_name)

    def save(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.graph, f, indent=2)

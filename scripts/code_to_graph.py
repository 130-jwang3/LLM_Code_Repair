"""
Script to convert code files into graph representations (e.g., AST, CPG).
"""
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import os
from itertools import count
import json

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

class ASTgraph:
    def __init__(self, source_code: bytes):
        self.source_code = source_code
        self.AST_graph = {"nodes": [], 
                          "edges": []}
        self.node_ids = count()

    def makeAST(self, root):
        stack = [(root, None)]

        while stack:
            node, parent_id = stack.pop()
            node_id = next(self.node_ids)

            self.AST_graph["nodes"].append({
                "id": node_id,
                "type": node.type,
                "start_byte": node.start_byte,
                "end_byte": node.end_byte,
                "start_point": node.start_point,
                "end_point": node.end_point,
                "text": self.source_code[node.start_byte:node.end_byte].decode("utf-8")
            })

            if parent_id is not None:
                self.AST_graph["edges"].append({
                    "parent": parent_id,
                    "child": node_id,
                    "type": "child"
                })

            for child in reversed(node.children):
                stack.append((child, node_id))

        return self.AST_graph


def code_to_graph(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    """
    Converts code in input_path to a graph (AST/CPG/etc) and saves to output_path.
    """
    with open(input_path, "rb") as file:
        source_code = file.read()

    tree = parser.parse(source_code)
    root_node = tree.root_node
    build = ASTgraph(source_code)
    ast_graph = build.build(root_node)


    #save to JSON
    with open(output_path, "w") as out:
        json.dump(ast_graph, out, indent = 2)


if __name__ == "__main__":
    code_to_graph("data/raw/example.py", "data/graphs/example_graph.json")
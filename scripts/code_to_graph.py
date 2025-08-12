import os
import sys
from .graph_builder import CodeGraphBuilder

def process_directory(input_dir: str, output_file: str, also_emit_ndjson: bool = True) -> None:
    """
    Walk a directory tree, parse all .py files, build the graph, and save:
      - output_file (JSON)
      - output_file.replace('.json', '.ndjson') (optional)
    """
    builder = CodeGraphBuilder()
    # Pass 1: register defs
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".py"):
                builder.first_pass(os.path.join(root, file))
    # Pass 2: resolve imports/calls
    builder.second_pass()
    builder.save(output_file)
    if also_emit_ndjson:
        ndjson = output_file[:-5] + ".ndjson" if output_file.endswith(".json") else output_file + ".ndjson"
        builder.save_stride_index(ndjson)

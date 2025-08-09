import os
import sys
from .graph_builder import CodeGraphBuilder

def process_directory(input_dir, output_file):
    builder = CodeGraphBuilder()
    # Pass 1: register defs
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".py"):
                builder.first_pass(os.path.join(root, file))
    # Pass 2: resolve imports/calls
    builder.second_pass()
    builder.save(output_file)


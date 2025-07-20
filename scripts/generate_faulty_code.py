import os
import sys
import subprocess
import ast
import astor
import shutil

# === Multiple Mutators ===

class MultiMutator(ast.NodeTransformer):
    def visit_Compare(self, node):
        self.generic_visit(node)
        for i, op in enumerate(node.ops):
            if isinstance(op, ast.Eq):
                node.ops[i] = ast.NotEq()
            elif isinstance(op, ast.NotEq):
                node.ops[i] = ast.Eq()
            elif isinstance(op, ast.Gt):
                node.ops[i] = ast.Lt()
            elif isinstance(op, ast.Lt):
                node.ops[i] = ast.Gt()
        return node

    def visit_NameConstant(self, node):
        if isinstance(node.value, bool):
            return ast.copy_location(ast.NameConstant(value=not node.value), node)
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Add):
            node.op = ast.Sub()
        elif isinstance(node.op, ast.Sub):
            node.op = ast.Add()
        elif isinstance(node.op, ast.Mult):
            node.op = ast.Div()
        elif isinstance(node.op, ast.Div):
            node.op = ast.Mult()
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.And):
            node.op = ast.Or()
        elif isinstance(node.op, ast.Or):
            node.op = ast.And()
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return node.operand  # remove not
        else:
            return ast.UnaryOp(op=ast.Not(), operand=node)

# === Mutation and File Logic ===

def mutate_file(src_path, dst_path, log_file):
    with open(src_path, 'r', encoding='utf-8') as f:
        code = f.read()
    try:
        tree = ast.parse(code)
    except Exception as e:
        print(f"Skipping {src_path}: {e}")
        return

    mutator = MultiMutator()
    mutated_tree = mutator.visit(tree)
    ast.fix_missing_locations(mutated_tree)
    mutated_code = astor.to_source(mutated_tree)

    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(mutated_code)

    log_file.write(f"{src_path}\n")

def clone_repo_if_needed(repo_url, clone_dir):
    if not os.path.exists(clone_dir):
        print(f"Cloning {repo_url} into {clone_dir}...")
        result = subprocess.run(['gh', 'repo', 'clone', repo_url, clone_dir])
        if result.returncode != 0:
            raise RuntimeError("Failed to clone the repository!")
    else:
        print(f"Repo already cloned at {clone_dir}, skipping clone.")

def generate_faulty_code(input_dir, output_dir):
    if os.path.exists(output_dir):
        print(f"Removing previous output directory: {output_dir}")
        try:
            shutil.rmtree(output_dir)
        except PermissionError:
            print("PermissionError while deleting. Retrying with ignore_errors=True...")
            shutil.rmtree(output_dir, ignore_errors=True)

    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "mutated_files.txt")
    with open(log_path, "w", encoding='utf-8') as log_file:
        for root, dirs, files in os.walk(input_dir):
            # Skip .git directories
            if '.git' in root.split(os.sep):
                continue

            rel_dir = os.path.relpath(root, input_dir)
            dst_root = os.path.join(output_dir, rel_dir) if rel_dir != '.' else output_dir
            os.makedirs(dst_root, exist_ok=True)

            is_test_dir = 'tests' in root.split(os.sep) or 'test' in root.split(os.sep)

            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_root, file)

                if file.endswith('.py') and not is_test_dir:
                    mutate_file(src_file, dst_file, log_file)
                else:
                    shutil.copy2(src_file, dst_file)

    print(f"\n✅ Mutation complete. Mutants saved to: {output_dir}")
    print(f"📝 Mutated files logged in: {log_path}")

if __name__ == "__main__":
    # Customize repo and paths
    repo_url = "PyGithub/PyGithub"
    repo_name = repo_url.split("/")[-1]
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    clone_dir = os.path.join(root_dir, "data", "raw", repo_name)
    mutants_dir = os.path.join(root_dir, "data", "processed", f"{repo_name}_mutants")

    clone_repo_if_needed(repo_url, clone_dir)
    generate_faulty_code(clone_dir, mutants_dir)

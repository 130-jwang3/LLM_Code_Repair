import os
import subprocess
def clone_repo_if_needed(repo_url, clone_dir):
    if not os.path.exists(clone_dir):
        print(f"Cloning {repo_url} into {clone_dir}...")
        result = subprocess.run(['gh', 'repo', 'clone', repo_url, clone_dir])
        if result.returncode != 0:
            raise RuntimeError("Failed to clone the repository!")
    else:
        print(f"Repo already cloned at {clone_dir}, skipping clone.")
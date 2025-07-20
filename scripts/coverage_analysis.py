"""
Script to analyze test coverage and associate it with code or graph nodes/edges.
"""

import coverage 
import pytest
import os

def analyze_coverage(source_dir, output_dir, repo_path):
    """
    Runs coverage analysis for code in source_dir and saves results to output_dir.
    """

    os.makedirs(output_dir, exist_ok=True)
    os.chdir(repo_path)

    cov = coverage.Coverage(source=source_dir)
    cov.start()

    pytest.main(["tests/"])

    cov.stop()
    cov.save()

    cov.json_report(directory=output_dir)



if __name__ == "__main__":
    analyze_coverage("data/raw/PyGithub/github", "data/coverage/PyGithub", "data/raw/PyGithub")
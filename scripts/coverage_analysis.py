import coverage
import pytest
import os
import shutil

def analyze_test_coverage(source_dir, output_dir, repo_path):
    os.makedirs(output_dir, exist_ok=True)

    # Absolute paths
    abs_output_path = os.path.abspath(output_dir)
    abs_repo_path = os.path.abspath(repo_path)

    os.chdir(abs_repo_path)

    cov = coverage.Coverage()
    cov.start()

    pytest.main(["tests/"])

    cov.stop()
    cov.save()

    # Move .coverage file to output_dir for clarity
    coverage_file = os.path.join(abs_repo_path, ".coverage")
    if os.path.exists(coverage_file):
        shutil.copy(coverage_file, os.path.join(abs_output_path, ".coverage"))

    # Re-load the saved .coverage file from output_dir to export JSON
    cov = coverage.Coverage(data_file=os.path.join(abs_output_path, ".coverage"))
    cov.load()
    cov.json_report(outfile=os.path.join(abs_output_path, "coverage.json"))

    print(f"[✓] Coverage report saved at {os.path.join(abs_output_path, 'coverage.json')}")

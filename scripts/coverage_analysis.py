# scripts/coverage_analysis.py
import coverage
import pytest
import os
import shutil
import json
from datetime import datetime

def analyze_test_coverage(source_dir, output_dir, repo_path):
    """
    Run coverage over pytest in the target repo, but be resilient:
      - If pytest/imports fail, we still save whatever coverage was collected.
      - If no tests or export fails, write a minimal stub JSON.
      - We never modify the upstream repo; data is copied to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    abs_output_path = os.path.abspath(output_dir)
    abs_repo_path = os.path.abspath(repo_path)

    prev_cwd = os.getcwd()
    try:
        os.chdir(abs_repo_path)

        # Help tests import the repo without installing
        os.environ["PYTHONPATH"] = abs_repo_path + os.pathsep + os.environ.get("PYTHONPATH", "")

        cov = coverage.Coverage(source=[abs_repo_path])
        cov.start()

        pytest_rc = None
        try:
            # Run tests only if a tests/ dir exists; be quiet and tolerant
            if os.path.isdir(os.path.join(abs_repo_path, "tests")):
                # You can tweak '-k' to skip slow/integration if needed
                pytest_rc = pytest.main(["-q", "--maxfail=1", "--disable-warnings", "tests/"])
            else:
                print("[info] No tests/ directory found; collecting zero coverage.")
        except SystemExit as e:
            pytest_rc = int(e.code)
        except Exception as e:
            print(f"[ALERT] Pytest raised an exception: {e}")

        cov.stop()
        cov.save()

        # Copy .coverage to output dir for clarity
        cov_data_in_repo = os.path.join(abs_repo_path, ".coverage")
        cov_data_out = os.path.join(abs_output_path, ".coverage")
        if os.path.exists(cov_data_in_repo):
            try:
                shutil.copy(cov_data_in_repo, cov_data_out)
            except Exception as e:
                print(f"[ALERT] Could not copy .coverage file: {e}")

        # Re-load from output_dir and try to export JSON
        json_path = os.path.join(abs_output_path, "coverage.json")
        try:
            cov2 = coverage.Coverage(data_file=cov_data_out if os.path.exists(cov_data_out) else None)
            cov2.load()
            cov2.json_report(outfile=json_path)
            print(f"[OK] Coverage report saved at {json_path}")
        except Exception as e:
            # Fallback: write a minimal stub so downstream never breaks
            print(f"[ALERT] coverage.json export failed: {e} — writing stub")
            stub = {
                "meta": {
                    "version": "stub",
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "note": "coverage export failed; this is an empty stub",
                },
                "files": {},
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(stub, f, indent=2)
            print(f"[OK] Stub coverage written at {json_path}")

        return {"pytest_returncode": pytest_rc, "coverage_json": json_path}

    finally:
        # Always restore working dir
        os.chdir(prev_cwd)

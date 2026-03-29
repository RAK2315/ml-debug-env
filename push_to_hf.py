# push_to_hf.py
# run from ml_debug_env/: python push_to_hf.py

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "rak2315/ml-debug-env"
REPO_TYPE = "space"

api = HfApi()

# Files to upload: (local_path, path_in_repo)
root = Path(__file__).parent

files = [
    (root / "models.py",              "models.py"),
    (root / "client.py",              "client.py"),
    (root / "__init__.py",            "__init__.py"),
    (root / "openenv.yaml",           "openenv.yaml"),
    (root / "README.md",              "README.md"),
    (root / "pyproject.toml",         "pyproject.toml"),
    (root / "server" / "app.py",               "server/app.py"),
    (root / "server" / "bug_generator.py",     "server/bug_generator.py"),
    (root / "server" / "grader.py",            "server/grader.py"),
    (root / "server" / "ml_debug_env_environment.py", "server/ml_debug_env_environment.py"),
    (root / "server" / "baseline_inference.py","server/baseline_inference.py"),
    (root / "server" / "requirements.txt",     "server/requirements.txt"),
    (root / "server" / "__init__.py",          "server/__init__.py"),
    (root / "server" / "Dockerfile",           "Dockerfile"),  # HF needs Dockerfile at root
]

print(f"Uploading to {REPO_ID}...\n")

for local_path, repo_path in files:
    if not local_path.exists():
        print(f"  SKIP (not found): {local_path}")
        continue
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"  OK: {repo_path}")
    except Exception as e:
        print(f"  FAIL: {repo_path} — {e}")

print("\nDone. Check https://huggingface.co/spaces/rak2315/ml-debug-env")

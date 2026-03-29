# test_hf.py
# run from ml_debug_env/: python test_hf.py

import subprocess
import sys
import os

print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")

# check huggingface_hub installed
try:
    import huggingface_hub
    print(f"huggingface_hub version: {huggingface_hub.__version__}")
except ImportError:
    print("huggingface_hub NOT installed")
    sys.exit(1)

# check what CLI scripts exist in venv
scripts_dir = os.path.join(os.path.dirname(sys.executable))
print(f"\nScripts dir: {scripts_dir}")
hf_scripts = [f for f in os.listdir(scripts_dir) if "hugging" in f.lower() or "hf" in f.lower()]
print(f"HF-related scripts found: {hf_scripts}")

# try login
print("\nAttempting HuggingFace login...")
print("Go to https://huggingface.co/settings/tokens and create a WRITE token")
print("Paste it below:\n")
from huggingface_hub import login
login()
print("\nLogin successful!")
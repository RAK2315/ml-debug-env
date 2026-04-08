# test_submission.py
# Run this before every submission to catch validator failures early.
# Usage: python test_submission.py

import os
import sys
import json
import subprocess
import urllib.request
import urllib.error
import time

LOCAL_URL = "http://localhost:8000"
HF_URL    = "https://rak2315-ml-debug-env.hf.space"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

results = []

def check(name, passed, detail=""):
    icon = PASS if passed else FAIL
    print(f"{icon} {name}" + (f" — {detail}" if detail else ""))
    results.append((name, passed))

# ── 1. inference.py exists ────────────────────────────────────────────────────
check("inference.py exists at repo root", os.path.exists("inference.py"))

# ── 2. baseline_inference.py reads API_BASE_URL / API_KEY ────────────────────
bi_path = os.path.join("server", "baseline_inference.py")
if os.path.exists(bi_path):
    content = open(bi_path).read()
    uses_api_base = "API_BASE_URL" in content
    uses_api_key  = "API_KEY" in content
    check("baseline_inference.py uses API_BASE_URL", uses_api_base)
    check("baseline_inference.py uses API_KEY", uses_api_key)
    check("baseline_inference.py does NOT hardcode Groq URL only",
          "API_BASE_URL" in content,
          "must prefer injected base_url over hardcoded Groq")
else:
    check("server/baseline_inference.py exists", False)

# ── 3. inference.py tries localhost first ─────────────────────────────────────
infer_content = open("inference.py").read() if os.path.exists("inference.py") else ""
localhost_pos = infer_content.find("localhost")
hf_pos        = infer_content.find("hf.space")
if localhost_pos != -1 and hf_pos != -1:
    check("inference.py tries localhost before HF Space", localhost_pos < hf_pos)
elif localhost_pos != -1:
    check("inference.py tries localhost", True)
else:
    check("inference.py tries localhost", False, "only HF Space URL found — validator can't reach it")

# ── 4. Run inference.py and check structured output ──────────────────────────
print("\n── Running inference.py ──")
env = os.environ.copy()
env["API_BASE_URL"] = os.environ.get("API_BASE_URL", "http://localhost:8000/v1")
env["API_KEY"]      = os.environ.get("API_KEY", "test-key")

proc = subprocess.run(
    [sys.executable, "inference.py"],
    capture_output=True, text=True, timeout=60, env=env
)
stdout = proc.stdout
stderr = proc.stderr

print("STDOUT:\n", stdout[:2000] if stdout else "(empty)")
if stderr:
    print("STDERR:\n", stderr[:500])

check("inference.py exits with code 0", proc.returncode == 0,
      f"exit code {proc.returncode}")
check("[START] found in stdout", "[START]" in stdout)
check("[STEP]  found in stdout", "[STEP]"  in stdout)
check("[END]   found in stdout", "[END]"   in stdout)

# Parse and validate blocks
tasks_found = []
for line in stdout.splitlines():
    if line.startswith("[END]"):
        parts = dict(p.split("=") for p in line[5:].strip().split() if "=" in p)
        tasks_found.append(parts)

check("At least 3 [END] blocks found", len(tasks_found) >= 3,
      f"found {len(tasks_found)}")

for t in tasks_found:
    tid   = t.get("task", "?")
    score = float(t.get("score", -1))
    check(f"Task {tid} score in [0.0, 1.0]", 0.0 <= score <= 1.0, f"score={score}")

# ── 5. Check local server is reachable ───────────────────────────────────────
print("\n── Checking local server ──")
try:
    with urllib.request.urlopen(f"{LOCAL_URL}/health", timeout=5) as r:
        body = json.loads(r.read())
    check("GET /health returns 200", True, str(body))
except Exception as e:
    check("GET /health reachable", False, str(e))
    print(f"  {WARN} Start your server: uvicorn server.app:app --host 0.0.0.0 --port 8000")

try:
    with urllib.request.urlopen(f"{LOCAL_URL}/tasks", timeout=5) as r:
        tasks = json.loads(r.read())
    check("GET /tasks returns task list", isinstance(tasks, (list, dict)), str(tasks)[:80])
except Exception as e:
    check("GET /tasks reachable", False, str(e))

# ── 6. Check /baseline endpoint ──────────────────────────────────────────────
print("\n── Checking /baseline endpoint ──")
try:
    with urllib.request.urlopen(f"{LOCAL_URL}/baseline", timeout=120) as r:
        data = json.loads(r.read())
    bres = data.get("results", [])
    avg  = data.get("average_score", 0)
    check("/baseline returns results list", len(bres) > 0, f"{len(bres)} tasks")
    check("/baseline average_score present", "average_score" in data, f"avg={avg:.3f}")
    for r in bres:
        check(f"  task {r['task_id']} score valid", 0.0 <= r['score'] <= 1.0,
              f"score={r['score']}")
except Exception as e:
    check("/baseline reachable", False, str(e))
    print(f"  {WARN} Make sure GROQ_API_KEY or API_KEY is set and server is running")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n══ SUMMARY ══════════════════════════════")
passed = sum(1 for _, p in results if p)
total  = len(results)
print(f"{passed}/{total} checks passed")
if passed == total:
    print(f"{PASS} Ready to submit!")
else:
    print(f"{FAIL} Fix the above before submitting.")
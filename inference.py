# inference.py
# Hackathon Phase 2 baseline inference script.
# Hits the deployed HF Space /baseline endpoint directly.
# Falls back to local server if HF Space is unreachable.

import os
import sys
import json
import urllib.request
import urllib.error

HF_SPACE_URL = "https://rak2315-ml-debug-env.hf.space"
LOCAL_URL = "http://localhost:8000"


def hit_baseline(base_url: str, timeout: int = 120) -> dict:
    url = f"{base_url}/baseline"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main():
    for base_url in [HF_SPACE_URL, LOCAL_URL]:
        try:
            print(f"Connecting to {base_url}/baseline ...", flush=True)
            data = hit_baseline(base_url)
            break
        except Exception as e:
            print(f"  Could not reach {base_url}: {e}", flush=True)
            data = None

    if data is None:
        print("ERROR: Could not reach any server endpoint.", file=sys.stderr)
        sys.exit(1)

    results = data.get("results", [])
    avg = data.get("average_score", 0.0)

    print("\n=== BASELINE RESULTS ===")
    for r in results:
        print(
            f"Task: {r['task_id']:<20} "
            f"Score: {r['score']:.1f}  "
            f"Bug type: {r['bug_type_submitted']}"
        )

    print(f"\nAverage score: {avg:.4f}")
    print(f"Model: {data.get('model', 'unknown')}")
    print("========================")

    with open("baseline_results.json", "w") as f:
        json.dump(data, f, indent=2)
    print("\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()
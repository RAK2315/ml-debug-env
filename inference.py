# inference.py
# Hackathon required baseline inference script.
# Runs the Groq-based baseline agent against all 3 tasks.
# Usage: python inference.py

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

from baseline_inference import run_baseline_on_all_tasks

if __name__ == "__main__":
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable not set.")
        sys.exit(1)

    print("Running baseline inference on all 3 tasks...\n")
    results = run_baseline_on_all_tasks(api_key)

    print("\n=== RESULTS ===")
    total = 0.0
    for r in results:
        print(f"Task: {r['task_id']} | Score: {r['score']} | Bug type: {r['bug_type_submitted']}")
        total += r["score"]

    print(f"\nAverage score: {total / len(results):.4f}")

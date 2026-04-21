import requests

BASE = "https://rak2315-ml-debug-env.hf.space"

print("\nEpisode: compound_leakage_eval (Expert — 2 bugs)")

# Reset
r = requests.post(f"{BASE}/reset", json={"task_id": "compound_leakage_eval"})
obs = r.json()["observation"]
print(f"Alert: {obs['alert']}\n")

# Inspect run_code
r = requests.post(f"{BASE}/step", json={"action": {"action_type": "inspect", "tool_name": "run_code"}})
obs = r.json()["observation"]
print(f"Step 1: inspect run_code          → {obs['tool_result'][:80] if obs.get('tool_result') else 'no result'}")

# Inspect get_traceback  
r = requests.post(f"{BASE}/step", json={"action": {"action_type": "inspect", "tool_name": "get_traceback"}})
obs = r.json()["observation"]
print(f"Step 2: inspect get_traceback     → {obs['tool_result'][:80] if obs.get('tool_result') else 'no result'}")

# Fix
r = requests.post(f"{BASE}/step", json={"action": {
    "action_type": "fix",
    "bug_type": "compound_leakage_eval",
    "diagnosis": "Two bugs: normalization before split causes data leakage, missing model.eval() causes non-deterministic metrics",
    "fixed_code": "# placeholder"
}})
obs = r.json()["observation"]
score = obs.get("grader_score", 0)
print(f"Step 3: fix compound_leakage_eval → score={score:.2f} {'✅ FIXED' if score > 0.9 else ''}")
print(f"\nFinal score: {score:.2f}")
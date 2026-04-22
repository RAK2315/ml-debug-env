import requests
import json
import time

BASE = "https://rak2315-ml-debug-env.hf.space"
session = requests.Session()

EPISODES = [
    {
        "task_id": "shape_mismatch",
        "label": "Episode 1: Shape Mismatch (Easy)",
        "steps": [
            {"action_type": "inspect", "tool_name": "run_code"},
            {"action_type": "inspect", "tool_name": "get_traceback"},
            {
                "action_type": "fix",
                "bug_type": "shape_mismatch",
                "diagnosis": "nn.Linear input dimension wrong — fc2 expects 128 but fc1 outputs 64",
                "fixed_code": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    x = torch.randn(32, 64)
    y = torch.randint(0, 10, (32,))
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, loss: {loss.item():.4f}")
print("Training finished")
""".strip(),
            },
        ],
    },
    {
        "task_id": "gradient_not_zeroed",
        "label": "Episode 2: Gradient Not Zeroed (Medium-Hard)",
        "steps": [
            {"action_type": "inspect", "tool_name": "inspect_gradients"},
            {
                "action_type": "fix",
                "bug_type": "gradient_not_zeroed",
                "diagnosis": "optimizer.zero_grad() missing before loss.backward() — gradients accumulate across batches causing explosion",
                "fixed_code": """
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(5):
    x = torch.randn(32, 16)
    y = torch.randn(32, 1)
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, loss: {loss.item():.4f}")
print("Training finished")
""".strip(),
            },
        ],
    },
    {
        "task_id": "compound_leakage_eval",
        "label": "Episode 3: Compound Leakage + Eval Mode (Expert — 2 bugs)",
        "steps": [
            {"action_type": "inspect", "tool_name": "run_code"},
            {"action_type": "inspect", "tool_name": "print_shapes"},
            {
                "action_type": "fix",
                "bug_type": "compound_leakage_eval",
                "diagnosis": "Two bugs: (1) normalization computed on full dataset before train/test split causes data leakage, (2) model.eval() missing during evaluation causes non-deterministic metrics due to active dropout",
                "fixed_code": """
import torch
import torch.nn as nn

torch.manual_seed(42)

X = torch.randn(200, 16)
y = (X[:, 0] > 0).float()

split = 160
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

mean = X_train.mean(dim=0)
std = X_train.std(dim=0) + 1e-8
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

model = nn.Sequential(nn.Linear(16, 32), nn.Dropout(0.3), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

model.train()
for epoch in range(5):
    optimizer.zero_grad()
    out = model(X_train).squeeze()
    loss = loss_fn(out, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    preds = (model(X_test).squeeze() > 0).float()
    acc = (preds == y_test).float().mean().item()
print(f"Accuracy: {acc:.4f}")
print("Evaluation complete")
""".strip(),
            },
        ],
    },
]


def print_separator(char="-", width=60):
    print(char * width)


def run_episode(episode):
    task_id = episode["task_id"]
    label = episode["label"]

    print_separator("=")
    print(f"  {label}")
    print_separator("=")

    r = session.post(f"{BASE}/reset", json={"task_id": task_id})
    if r.status_code != 200:
        print(f"  Reset failed: {r.status_code}")
        return
    obs = r.json()["observation"]
    print(f"\n  Alert: \"{obs['alert']}\"")
    print(f"  Tools: {obs['available_tools']}")
    print(f"  Step budget: {obs['step_budget']} | Bugs: {obs['num_bugs']}")
    print()

    for i, step in enumerate(episode["steps"]):
        action_type = step["action_type"]
        time.sleep(0.5)

        if action_type == "inspect":
            tool = step["tool_name"]
            r = session.post(f"{BASE}/step", json={"action": step})
            if r.status_code != 200:
                print(f"  Step {i+1}: inspect:{tool} → ERROR {r.status_code}")
                continue
            obs = r.json()["observation"]
            result = obs.get("tool_result", "") or ""
            preview = result[:120].replace("\n", " ").strip()
            budget = obs.get("step_budget", "?")
            print(f"  Step {i+1}: inspect:{tool:<20} reward=+0.00  budget={budget}")
            print(f"           → {preview}...")
            print()

        elif action_type == "fix":
            r = session.post(f"{BASE}/step", json={"action": step})
            if r.status_code != 200:
                print(f"  Step {i+1}: fix → ERROR {r.status_code}")
                continue
            obs = r.json()["observation"]
            score = obs.get("grader_score", 0) or 0
            feedback = obs.get("grader_feedback", "") or ""
            multiplier = obs.get("efficiency_multiplier", 1.0) or 1.0
            budget = obs.get("step_budget", "?")
            status = "✅ FIXED" if score >= 0.95 else ("🟡 PARTIAL" if score >= 0.6 else "❌")
            print(f"  Step {i+1}: fix:{step['bug_type']:<25} reward={score:.2f}  {status}")
            if multiplier > 1.0:
                print(f"           → Efficiency bonus: ×{multiplier} applied")
            print(f"           → {feedback[:120]}")
            print()

    print()


def main():
    print()
    print("=" * 60)
    print("  ML Debug Env — Live Demo")
    print("  Agent debugs broken PyTorch scripts using tool calls")
    print("  Partial observability: alert only on reset, no code")
    print("=" * 60)
    print()

    print(f"  Environment: {BASE}")
    r = session.get(f"{BASE}/health")
    print(f"  Health: {r.json()}")
    print()

    for episode in EPISODES:
        run_episode(episode)
        time.sleep(1)

    print_separator("=")
    print("  Demo complete.")
    print("  Scoring ladder: 0.01 → wrong type | 0.20 → crashes")
    print("                  0.40 → incomplete | 0.60 → not fixed")
    print("                  0.80 → missing signal | 0.99 → perfect ✅")
    print_separator("=")
    print()


if __name__ == "__main__":
    main()
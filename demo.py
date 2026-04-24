import requests
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
                "bug_type": "wrong_device",
                "diagnosis": "Looks like a device mismatch error",
                "fixed_code": "# wrong fix\nimport torch\nmodel = model.to('cpu')",
            },
            {
                "action_type": "fix",
                "bug_type": "shape_mismatch",
                "diagnosis": "nn.Linear input dimension wrong — fc2 expects 128 but fc1 outputs 64",
                "fixed_code": (
                    "import torch\nimport torch.nn as nn\n\n"
                    "class Model(nn.Module):\n"
                    "    def __init__(self):\n"
                    "        super().__init__()\n"
                    "        self.fc1 = nn.Linear(64, 128)\n"
                    "        self.fc2 = nn.Linear(128, 10)\n"
                    "    def forward(self, x):\n"
                    "        x = torch.relu(self.fc1(x))\n"
                    "        return self.fc2(x)\n\n"
                    "model = Model()\n"
                    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n"
                    "loss_fn = nn.CrossEntropyLoss()\n\n"
                    "for epoch in range(3):\n"
                    "    x = torch.randn(32, 64)\n"
                    "    y = torch.randint(0, 10, (32,))\n"
                    "    optimizer.zero_grad()\n"
                    "    out = model(x)\n"
                    "    loss = loss_fn(out, y)\n"
                    "    loss.backward()\n"
                    "    optimizer.step()\n"
                    "    print(f'Epoch {epoch+1}, loss: {loss.item():.4f}')\n"
                    "print('Training finished')"
                ),
            },
        ],
    },
    {
        "task_id": "compound_leakage_eval",
        "label": "Episode 2: Compound Leakage + Eval Mode (Expert — 2 bugs)",
        "steps": [
            {"action_type": "inspect", "tool_name": "run_code"},
            {"action_type": "inspect", "tool_name": "inspect_gradients"},
            {
                "action_type": "fix",
                "bug_type": "data_leakage",
                "diagnosis": "Only data leakage fixed — normalization before split",
                "fixed_code": (
                    "import torch\nimport torch.nn as nn\n\n"
                    "torch.manual_seed(42)\n"
                    "X = torch.randn(200, 16)\n"
                    "y = (X[:, 0] > 0).float()\n"
                    "split = 160\n"
                    "X_train, X_test = X[:split], X[split:]\n"
                    "y_train, y_test = y[:split], y[split:]\n"
                    "mean = X_train.mean(dim=0)\n"
                    "std = X_train.std(dim=0) + 1e-8\n"
                    "X_train = (X_train - mean) / std\n"
                    "X_test = (X_test - mean) / std\n"
                    "model = nn.Sequential(nn.Linear(16,32), nn.Dropout(0.3), nn.ReLU(), nn.Linear(32,1))\n"
                    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n"
                    "loss_fn = nn.BCEWithLogitsLoss()\n"
                    "model.train()\n"
                    "for epoch in range(5):\n"
                    "    optimizer.zero_grad()\n"
                    "    out = model(X_train).squeeze()\n"
                    "    loss = loss_fn(out, y_train)\n"
                    "    loss.backward()\n"
                    "    optimizer.step()\n"
                    "    print(f'Epoch {epoch+1}, loss: {loss.item():.4f}')\n"
                    "preds = (model(X_test).squeeze() > 0).float()\n"
                    "acc = (preds == y_test).float().mean().item()\n"
                    "print(f'Accuracy: {acc:.4f}')\n"
                    "print('Evaluation complete')"
                ),
            },
            {
                "action_type": "fix",
                "bug_type": "compound_leakage_eval",
                "diagnosis": "Two bugs: normalization on full dataset before split (leakage) + model.eval() missing during eval (dropout active)",
                "fixed_code": (
                    "import torch\nimport torch.nn as nn\n\n"
                    "torch.manual_seed(42)\n"
                    "X = torch.randn(200, 16)\n"
                    "y = (X[:, 0] > 0).float()\n"
                    "split = 160\n"
                    "X_train, X_test = X[:split], X[split:]\n"
                    "y_train, y_test = y[:split], y[split:]\n"
                    "mean = X_train.mean(dim=0)\n"
                    "std = X_train.std(dim=0) + 1e-8\n"
                    "X_train = (X_train - mean) / std\n"
                    "X_test = (X_test - mean) / std\n"
                    "model = nn.Sequential(nn.Linear(16,32), nn.Dropout(0.3), nn.ReLU(), nn.Linear(32,1))\n"
                    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n"
                    "loss_fn = nn.BCEWithLogitsLoss()\n"
                    "model.train()\n"
                    "for epoch in range(5):\n"
                    "    optimizer.zero_grad()\n"
                    "    out = model(X_train).squeeze()\n"
                    "    loss = loss_fn(out, y_train)\n"
                    "    loss.backward()\n"
                    "    optimizer.step()\n"
                    "    print(f'Epoch {epoch+1}, loss: {loss.item():.4f}')\n"
                    "model.eval()\n"
                    "with torch.no_grad():\n"
                    "    preds = (model(X_test).squeeze() > 0).float()\n"
                    "    acc = (preds == y_test).float().mean().item()\n"
                    "print(f'Accuracy: {acc:.4f}')\n"
                    "print('Evaluation complete')"
                ),
            },
        ],
    },
]


def sep(char="-", w=62):
    print(char * w)


def run_episode(episode):
    sep("=")
    print(f"  {episode['label']}")
    sep("=")

    r = session.post(f"{BASE}/reset", json={"task_id": episode["task_id"]})
    if r.status_code != 200:
        print(f"  Reset failed: {r.status_code}")
        return
    obs = r.json()["observation"]
    print(f"\n  Alert: \"{obs['alert']}\"")
    print(f"  Bugs: {obs['num_bugs']} | Step budget: {obs['step_budget']}\n")

    step_num = 0
    for step in episode["steps"]:
        time.sleep(0.5)
        step_num += 1
        atype = step["action_type"]

        if atype == "inspect":
            tool = step["tool_name"]
            r = session.post(f"{BASE}/step", json={"action": step})
            if r.status_code != 200:
                print(f"  Step {step_num}: inspect:{tool} → ERROR {r.status_code}")
                continue
            obs = r.json()["observation"]
            result = (obs.get("tool_result") or "").replace("\n", " ").strip()[:100]
            budget = obs.get("step_budget", "?")
            print(f"  Step {step_num}: inspect:{tool:<22} reward= 0.00  budget={budget}")
            print(f"           → {result}...")
            print()

        elif atype == "fix":
            r = session.post(f"{BASE}/step", json={"action": step})
            if r.status_code != 200:
                print(f"  Step {step_num}: fix → ERROR {r.status_code}")
                continue
            obs = r.json()["observation"]
            score = obs.get("grader_score") or 0
            feedback = (obs.get("grader_feedback") or "")[:100]
            multiplier = obs.get("efficiency_multiplier") or 1.0

            if score >= 0.95:
                status = "← FIXED ✅"
            elif score >= 0.6:
                status = "← partial"
            elif score >= 0.2:
                status = "← wrong fix"
            else:
                status = "← wrong type"

            print(f"  Step {step_num}: fix:{step['bug_type']:<26} reward={score:.2f}  {status}")
            if multiplier > 1.0 and score >= 0.95:
                print(f"           → Efficiency bonus: x{multiplier} applied")
            print(f"           → {feedback}")
            print()

            if obs.get("done") and score >= 0.95:
                break

    print()


def main():
    print()
    sep("=")
    print("  ML Debug Env — Live Demo")
    print("  Agent debugs broken PyTorch scripts using tool calls")
    print("  Partial observability: alert only, no code on reset")
    sep("=")
    print()
    r = session.get(f"{BASE}/health")
    print(f"  Environment: {BASE}")
    print(f"  Health: {r.json()}\n")

    for ep in EPISODES:
        run_episode(ep)
        time.sleep(1)

    sep("=")
    print("  Demo complete.")
    print("  Scoring: 0.01 wrong type | 0.20 crashes | 0.40 incomplete")
    print("           0.60 not fixed  | 0.80 no signal | 0.99 perfect ✅")
    sep("=")
    print()


if __name__ == "__main__":
    main()
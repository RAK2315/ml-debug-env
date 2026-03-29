# server/bug_generator.py
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class BugScenario:
    task_id: str
    task_description: str
    buggy_code: str
    error_output: str
    correct_bug_type: str
    solution_hint: str  # used by grader to verify fix direction, not shown to agent


TASK_SHAPE_MISMATCH = "shape_mismatch"
TASK_TRAINING_COLLAPSE = "training_collapse"
TASK_DATA_LEAKAGE = "data_leakage"


def get_scenario(task_id: str, seed: Optional[int] = None) -> BugScenario:
    rng = random.Random(seed)
    if task_id == TASK_SHAPE_MISMATCH:
        return _shape_mismatch_scenario(rng)
    elif task_id == TASK_TRAINING_COLLAPSE:
        return _training_collapse_scenario(rng)
    elif task_id == TASK_DATA_LEAKAGE:
        return _data_leakage_scenario(rng)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")


def get_random_task(seed: Optional[int] = None) -> str:
    rng = random.Random(seed)
    return rng.choice([TASK_SHAPE_MISMATCH, TASK_TRAINING_COLLAPSE, TASK_DATA_LEAKAGE])


# ──────────────────────────────────────────────────────────────
# TASK 1 — Shape Mismatch (Easy)
# The classifier head has wrong input features.
# Error is explicit in the traceback.
# ──────────────────────────────────────────────────────────────

def _shape_mismatch_scenario(rng: random.Random) -> BugScenario:
    hidden_size = rng.choice([128, 256, 512])
    wrong_size = rng.choice([64, 32, 16])
    num_classes = rng.choice([10, 5, 20])

    buggy_code = f'''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, {hidden_size}),
            nn.ReLU(),
            nn.Linear({hidden_size}, {hidden_size}),
            nn.ReLU(),
        )
        self.classifier = nn.Linear({wrong_size}, {num_classes})

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

X = torch.randn(200, 784)
y = torch.randint(0, {num_classes}, (200,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SimpleClassifier()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {{epoch+1}} complete")

print("Training finished")
'''

    error_output = f'''Traceback (most recent call last):
  File "train.py", line 32, in <module>
    pred = model(xb)
  File ".../torch/nn/modules/module.py", in _call_impl
    return forward(*input, **kwargs)
  File "train.py", line 21, in forward
    return self.classifier(features)
  File ".../torch/nn/modules/linear.py", in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied ({hidden_size} cannot be broadcast to {wrong_size})'''

    return BugScenario(
        task_id=TASK_SHAPE_MISMATCH,
        task_description=(
            "This PyTorch classifier crashes immediately during the forward pass. "
            "The training loop never completes a single step. "
            "Find the architectural bug and fix the script so it trains for 3 epochs without error."
        ),
        buggy_code=buggy_code,
        error_output=error_output,
        correct_bug_type="shape_mismatch",
        solution_hint=f"classifier input must be {hidden_size} not {wrong_size}",
    )


# ──────────────────────────────────────────────────────────────
# TASK 2 — Training Collapse (Medium)
# Loss explodes to NaN within a few steps.
# Script runs but model never learns. No crash, just broken behavior.
# ──────────────────────────────────────────────────────────────

def _training_collapse_scenario(rng: random.Random) -> BugScenario:
    bad_lr = rng.choice([10.0, 50.0, 100.0])
    variant = rng.choice(["lr", "loss_fn"])

    if variant == "lr":
        buggy_code = f'''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

X = torch.randn(300, 20)
y = (X[:, 0] + X[:, 1] * 0.5 + torch.randn(300) * 0.1)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32)

model = MLP()
optimizer = optim.SGD(model.parameters(), lr={bad_lr})
criterion = nn.MSELoss()

for epoch in range(5):
    epoch_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg = epoch_loss / len(loader)
    print(f"Epoch {{epoch+1}}, loss: {{avg:.4f}}")

print("Training finished")
'''
        error_output = (
            f"Training runs without crashing but loss diverges to NaN by epoch 2.\n"
            f"Output observed:\n"
            f"  Epoch 1, loss: 847.3291\n"
            f"  Epoch 2, loss: nan\n"
            f"  Epoch 3, loss: nan\n"
            f"  Epoch 4, loss: nan\n"
            f"  Epoch 5, loss: nan\n"
            f"The model produces NaN outputs and never learns the regression target."
        )
        solution_hint = f"learning rate {bad_lr} causes gradient explosion; reduce to ~1e-3"

    else:
        buggy_code = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

X = torch.randn(400, 15)
y = (X[:, 0] > 0).float()
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32)

model = BinaryClassifier()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(5):
    epoch_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg = epoch_loss / len(loader)
    print(f"Epoch {epoch+1}, loss: {avg:.4f}")

print("Training finished")
'''
        error_output = (
            "Training runs without error but model fails to converge.\n"
            "Output observed:\n"
            "  Epoch 1, loss: 0.2489\n"
            "  Epoch 2, loss: 0.2491\n"
            "  Epoch 3, loss: 0.2490\n"
            "  Epoch 4, loss: 0.2492\n"
            "  Epoch 5, loss: 0.2491\n"
            "Loss plateaus immediately and does not decrease. "
            "The model is using the wrong loss function for the task type."
        )
        solution_hint = "MSELoss used for binary classification; should be BCELoss or BCEWithLogitsLoss (remove sigmoid and use BCEWithLogitsLoss)"

    return BugScenario(
        task_id=TASK_TRAINING_COLLAPSE,
        task_description=(
            "This PyTorch training script runs without any Python errors or crashes, "
            "but the model completely fails to learn. "
            "Diagnose why training is broken and fix the script so loss decreases "
            "consistently across 5 epochs."
        ),
        buggy_code=buggy_code,
        error_output=error_output,
        correct_bug_type="training_collapse",
        solution_hint=solution_hint,
    )


# ──────────────────────────────────────────────────────────────
# TASK 3 — Data Leakage (Hard)
# Test set statistics used to normalize the entire dataset before splitting.
# Model trains, accuracy looks great (~95%), but it's artificially inflated.
# Agent must reason about data flow, not errors or loss curves.
# ──────────────────────────────────────────────────────────────

def _data_leakage_scenario(rng: random.Random) -> BugScenario:
    variant = rng.choice(["normalize_before_split", "scaler_fit_on_all"])

    if variant == "normalize_before_split":
        buggy_code = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class TabularClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)

torch.manual_seed(42)
N, D, C = 1000, 20, 3
X_raw = torch.randn(N, D)
true_weights = torch.randn(D, C)
y_all = (X_raw @ true_weights).argmax(dim=1)

# BUG: normalizing using statistics from the full dataset before splitting
# This leaks test set information into the training process
mean = X_raw.mean(dim=0)
std = X_raw.std(dim=0) + 1e-8
X_normalized = (X_raw - mean) / std

split = int(0.8 * N)
X_train, X_test = X_normalized[:split], X_normalized[split:]
y_train, y_test = y_all[:split], y_all[split:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

model = TabularClassifier(D, C)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    test_preds = model(X_test).argmax(dim=1)
    accuracy = (test_preds == y_test).float().mean().item()

print(f"Test accuracy: {accuracy:.4f}")
print("Training finished")
'''
        error_output = (
            "Script runs to completion with no errors.\n"
            "Reported test accuracy: 0.9650\n"
            "\n"
            "However, the reported test accuracy is misleading and cannot be trusted. "
            "The model has not demonstrated genuine generalization ability. "
            "There is a data handling bug that makes the evaluation invalid. "
            "The model would perform significantly worse on truly unseen data."
        )
        solution_hint = "normalize using only train set mean/std; compute mean and std after the split, only on X_train"

    else:
        buggy_code = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

N = 800
X_raw = torch.randn(N, 10)
y_all = X_raw[:, 0] * 2.5 + X_raw[:, 2] * 1.3 + torch.randn(N) * 0.3

split = int(0.75 * N)
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
y_train, y_test = y_all[:split], y_all[split:]

# BUG: computing normalization statistics from the full dataset
# The test set has already influenced mean and std
full_mean = X_raw.mean(dim=0)
full_std = X_raw.std(dim=0) + 1e-8
X_train = (X_train_raw - full_mean) / full_std
X_test = (X_test_raw - full_mean) / full_std

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

model = Regressor()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(8):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    test_loss = criterion(model(X_test), y_test).item()

print(f"Test MSE: {test_loss:.4f}")
print("Training finished")
'''
        error_output = (
            "Script runs to completion with no errors.\n"
            "Reported test MSE: 0.1021\n"
            "\n"
            "The reported test MSE is artificially low and cannot be trusted. "
            "There is a data preprocessing bug that leaks information from the test set "
            "into the normalization step. The evaluation does not reflect true generalization. "
            "A correctly implemented version would report higher test MSE."
        )
        solution_hint = "fit normalization stats only on X_train_raw; use train_mean and train_std to normalize both train and test"

    return BugScenario(
        task_id=TASK_DATA_LEAKAGE,
        task_description=(
            "This PyTorch training script runs cleanly with no errors and reports impressive metrics. "
            "But the evaluation is fundamentally broken due to a data handling mistake. "
            "The model has not learned to generalize — the good numbers are an illusion. "
            "Identify the data pipeline bug and fix it so the evaluation is valid."
        ),
        buggy_code=buggy_code,
        error_output=error_output,
        correct_bug_type="data_leakage",
        solution_hint=solution_hint,
    )
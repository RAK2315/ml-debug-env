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
    solution_hint: str


TASK_SHAPE_MISMATCH = "shape_mismatch"
TASK_TRAINING_COLLAPSE = "training_collapse"
TASK_DATA_LEAKAGE = "data_leakage"
TASK_WRONG_DEVICE = "wrong_device"
TASK_GRADIENT_NOT_ZEROED = "gradient_not_zeroed"
TASK_MISSING_EVAL_MODE = "missing_eval_mode"

ALL_TASKS = [
    TASK_SHAPE_MISMATCH,
    TASK_TRAINING_COLLAPSE,
    TASK_DATA_LEAKAGE,
    TASK_WRONG_DEVICE,
    TASK_GRADIENT_NOT_ZEROED,
    TASK_MISSING_EVAL_MODE,
]


def get_scenario(task_id: str, seed: Optional[int] = None) -> BugScenario:
    rng = random.Random(seed)
    if task_id == TASK_SHAPE_MISMATCH:
        return _shape_mismatch_scenario(rng)
    elif task_id == TASK_TRAINING_COLLAPSE:
        return _training_collapse_scenario(rng)
    elif task_id == TASK_DATA_LEAKAGE:
        return _data_leakage_scenario(rng)
    elif task_id == TASK_WRONG_DEVICE:
        return _wrong_device_scenario(rng)
    elif task_id == TASK_GRADIENT_NOT_ZEROED:
        return _gradient_not_zeroed_scenario(rng)
    elif task_id == TASK_MISSING_EVAL_MODE:
        return _missing_eval_mode_scenario(rng)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")


def get_random_task(seed: Optional[int] = None) -> str:
    rng = random.Random(seed)
    return rng.choice(ALL_TASKS)


# ──────────────────────────────────────────────────────────────
# TASK 1 — Shape Mismatch (Easy)
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
        solution_hint = "MSELoss used for binary classification; should be BCELoss or BCEWithLogitsLoss"

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
            "There is a data handling bug that makes the evaluation invalid."
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
            "into the normalization step."
        )
        solution_hint = "fit normalization stats only on X_train_raw; use train_mean and train_std to normalize both train and test"

    return BugScenario(
        task_id=TASK_DATA_LEAKAGE,
        task_description=(
            "This PyTorch training script runs cleanly with no errors and reports impressive metrics. "
            "But the evaluation is fundamentally broken due to a data handling mistake. "
            "Identify the data pipeline bug and fix it so the evaluation is valid."
        ),
        buggy_code=buggy_code,
        error_output=error_output,
        correct_bug_type="data_leakage",
        solution_hint=solution_hint,
    )


# ──────────────────────────────────────────────────────────────
# TASK 4 — Wrong Device (Medium)
# Model on CUDA, data on CPU (or vice versa). Explicit crash.
# Agent must identify device mismatch and fix tensor placement.
# ──────────────────────────────────────────────────────────────

def _wrong_device_scenario(rng: random.Random) -> BugScenario:
    hidden = rng.choice([64, 128, 256])
    num_classes = rng.choice([5, 10])

    buggy_code = f'''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, {hidden}),
            nn.ReLU(),
            nn.Linear({hidden}, {num_classes}),
        )

    def forward(self, x):
        return self.net(x)

X = torch.randn(200, 784)
y = torch.randint(0, {num_classes}, (200,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)
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

    error_output = (
        "Traceback (most recent call last):\n"
        "  File \"train.py\", line 30, in <module>\n"
        "    pred = model(xb)\n"
        "  File \".../torch/nn/modules/linear.py\", in forward\n"
        "    return F.linear(input, self.weight, self.bias)\n"
        "RuntimeError: Expected all tensors to be on the same device, "
        "but found at least two devices, cuda:0 and cpu!\n\n"
        "The model was moved to GPU but the data batches remain on CPU. "
        "Every forward pass crashes with a device mismatch error."
    )

    return BugScenario(
        task_id=TASK_WRONG_DEVICE,
        task_description=(
            "This PyTorch training script crashes on the first forward pass with a device mismatch error. "
            "The model and data tensors are on different devices. "
            "Fix the script so training runs for 3 epochs without error on whatever device is available."
        ),
        buggy_code=buggy_code,
        error_output=error_output,
        correct_bug_type="wrong_device",
        solution_hint="move xb and yb to device inside the training loop: xb, yb = xb.to(device), yb.to(device)",
    )


# ──────────────────────────────────────────────────────────────
# TASK 5 — Gradient Not Zeroed (Medium-Hard)
# optimizer.zero_grad() is missing. Gradients accumulate across
# batches, loss behaves erratically, model fails to converge.
# No crash. Agent must reason about the training loop structure.
# ──────────────────────────────────────────────────────────────

def _gradient_not_zeroed_scenario(rng: random.Random) -> BugScenario:
    hidden = rng.choice([32, 64, 128])
    lr = rng.choice([1e-3, 5e-4])

    buggy_code = f'''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, {hidden}),
            nn.ReLU(),
            nn.Linear({hidden}, {hidden}),
            nn.ReLU(),
            nn.Linear({hidden}, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

X = torch.randn(500, 10)
y = X[:, 0] * 1.5 - X[:, 2] * 0.8 + torch.randn(500) * 0.2
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = MLP()
optimizer = optim.Adam(model.parameters(), lr={lr})
criterion = nn.MSELoss()

for epoch in range(6):
    epoch_loss = 0.0
    for xb, yb in loader:
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
        "Script runs without crashing but training is highly unstable.\n"
        "Output observed:\n"
        "  Epoch 1, loss: 12.4821\n"
        "  Epoch 2, loss: 847.2341\n"
        "  Epoch 3, loss: 23451.8821\n"
        "  Epoch 4, loss: nan\n"
        "  Epoch 5, loss: nan\n"
        "  Epoch 6, loss: nan\n"
        "Loss explodes dramatically after the first epoch and collapses to NaN. "
        "No crash occurs. The model never converges. "
        "There is a fundamental error in the training loop structure."
    )

    return BugScenario(
        task_id=TASK_GRADIENT_NOT_ZEROED,
        task_description=(
            "This PyTorch training script runs without crashing but loss explodes "
            "after the first epoch and collapses to NaN. The model never learns anything. "
            "Find the training loop bug and fix it so loss decreases consistently across 6 epochs."
        ),
        buggy_code=buggy_code,
        error_output=error_output,
        correct_bug_type="gradient_not_zeroed",
        solution_hint="optimizer.zero_grad() is missing before loss.backward(); gradients accumulate across batches causing explosion",
    )


# ──────────────────────────────────────────────────────────────
# TASK 6 — Missing Eval Mode (Hard)
# model.eval() and torch.no_grad() missing during evaluation.
# Dropout active, BatchNorm uses batch stats not running stats.
# Everything runs. Metrics are noisy and unreliable. No crash.
# Agent must understand train vs eval mode semantics.
# ──────────────────────────────────────────────────────────────

def _missing_eval_mode_scenario(rng: random.Random) -> BugScenario:
    dropout_p = rng.choice([0.3, 0.4, 0.5])
    hidden = rng.choice([64, 128])
    num_classes = rng.choice([3, 5])

    buggy_code = f'''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class DropoutClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, {hidden}),
            nn.BatchNorm1d({hidden}),
            nn.ReLU(),
            nn.Dropout(p={dropout_p}),
            nn.Linear({hidden}, {hidden}),
            nn.BatchNorm1d({hidden}),
            nn.ReLU(),
            nn.Dropout(p={dropout_p}),
            nn.Linear({hidden}, num_classes),
        )

    def forward(self, x):
        return self.net(x)

torch.manual_seed(42)
N, D, C = 800, 20, {num_classes}
X = torch.randn(N, D)
true_w = torch.randn(D, C)
y = (X @ true_w).argmax(dim=1)

split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

model = DropoutClassifier(D, C)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {{epoch+1}} complete")

preds = model(X_test).argmax(dim=1)
accuracy = (preds == y_test).float().mean().item()
print(f"Test accuracy: {{accuracy:.4f}}")
print("Evaluation complete")
print("Training finished")
'''

    error_output = (
        "Script runs to completion with no errors.\n"
        f"Reported test accuracy varies between runs: 0.51, 0.58, 0.49, 0.61\n"
        "\n"
        f"The model has Dropout(p={dropout_p}) and BatchNorm layers. "
        "Test accuracy is inconsistent and significantly lower than expected. "
        "Running the same script multiple times gives different accuracy values each time. "
        "The evaluation is unreliable. No exception is raised. "
        "The model appears to be in the wrong mode during evaluation."
    )

    return BugScenario(
        task_id=TASK_MISSING_EVAL_MODE,
        task_description=(
            "This PyTorch classifier trains successfully but produces unreliable and "
            "inconsistent test accuracy. Running evaluation multiple times gives different results. "
            "The model has Dropout and BatchNorm layers. "
            "Fix the evaluation so it produces stable, correct metrics."
        ),
        buggy_code=buggy_code,
        error_output=error_output,
        correct_bug_type="missing_eval_mode",
        solution_hint=f"model.eval() and torch.no_grad() must be called before evaluation; dropout p={dropout_p} stays active in train mode causing stochastic predictions",
    )
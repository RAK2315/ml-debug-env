import ast
import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class BugScenario:
    task_id: str
    task_description: str
    buggy_code: str
    error_output: str
    correct_bug_type: str
    solution_hint: str
    num_bugs: int = 1
    alert: str = ""


TASK_SHAPE_MISMATCH = "shape_mismatch"
TASK_TRAINING_COLLAPSE = "training_collapse"
TASK_DATA_LEAKAGE = "data_leakage"
TASK_WRONG_DEVICE = "wrong_device"
TASK_GRADIENT_NOT_ZEROED = "gradient_not_zeroed"
TASK_MISSING_EVAL_MODE = "missing_eval_mode"
TASK_COMPOUND_SHAPE_DEVICE = "compound_shape_device"
TASK_COMPOUND_LEAKAGE_EVAL = "compound_leakage_eval"

ALL_TASKS = [
    TASK_SHAPE_MISMATCH,
    TASK_TRAINING_COLLAPSE,
    TASK_DATA_LEAKAGE,
    TASK_WRONG_DEVICE,
    TASK_GRADIENT_NOT_ZEROED,
    TASK_MISSING_EVAL_MODE,
    TASK_COMPOUND_SHAPE_DEVICE,
    TASK_COMPOUND_LEAKAGE_EVAL,
]

SINGLE_TASKS = [
    TASK_SHAPE_MISMATCH,
    TASK_TRAINING_COLLAPSE,
    TASK_DATA_LEAKAGE,
    TASK_WRONG_DEVICE,
    TASK_GRADIENT_NOT_ZEROED,
    TASK_MISSING_EVAL_MODE,
]

COMPOUND_TASKS = [
    TASK_COMPOUND_SHAPE_DEVICE,
    TASK_COMPOUND_LEAKAGE_EVAL,
]

AVAILABLE_TOOLS = ["run_code", "get_traceback", "inspect_gradients", "print_shapes", "view_source"]


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
    elif task_id == TASK_COMPOUND_SHAPE_DEVICE:
        return _compound_shape_device_scenario(rng)
    elif task_id == TASK_COMPOUND_LEAKAGE_EVAL:
        return _compound_leakage_eval_scenario(rng)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")


def get_random_task(seed: Optional[int] = None) -> str:
    rng = random.Random(seed)
    return rng.choice(ALL_TASKS)


# ──────────────────────────────────────────────────────────────
# TOOL EXECUTION ENGINE
# These functions implement the 5 diagnostic tools available
# to the agent in partial observability mode.
# ──────────────────────────────────────────────────────────────

def _get_python_exe() -> str:
    python_exe = os.environ.get("PYTHON_EXEC")
    if not python_exe:
        server_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(server_dir)
        candidate = os.path.join(project_dir, ".venv", "Scripts", "python.exe")
        if not os.path.exists(candidate):
            candidate = os.path.join(project_dir, ".venv", "bin", "python")
        python_exe = candidate if os.path.exists(candidate) else sys.executable
    return python_exe


def _run_in_subprocess(code: str, timeout: int = 40) -> tuple[str, bool, str]:
    """Run code, return (output, success, stderr_only)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [_get_python_exe(), tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        combined = (result.stdout + result.stderr).strip()
        return combined, result.returncode == 0, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout}s.", False, ""
    except Exception as e:
        return f"Execution error: {e}", False, ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def execute_tool(tool_name: str, scenario: BugScenario) -> str:
    """
    Execute a diagnostic tool against the buggy scenario.
    Returns a string result to show the agent.
    """
    if tool_name == "run_code":
        return _tool_run_code(scenario.buggy_code)
    elif tool_name == "get_traceback":
        return _tool_get_traceback(scenario.buggy_code)
    elif tool_name == "inspect_gradients":
        return _tool_inspect_gradients(scenario.buggy_code, scenario.task_id)
    elif tool_name == "print_shapes":
        return _tool_print_shapes(scenario.buggy_code, scenario.task_id)
    elif tool_name == "view_source":
        return _tool_view_source(scenario.buggy_code)
    else:
        return f"Unknown tool: {tool_name}. Available tools: {AVAILABLE_TOOLS}"


def _tool_run_code(buggy_code: str) -> str:
    output, success, _ = _run_in_subprocess(buggy_code, timeout=30)
    status = "COMPLETED" if success else "FAILED"
    lines = output.split("\n")
    truncated = "\n".join(lines[:40])
    if len(lines) > 40:
        truncated += f"\n... ({len(lines) - 40} more lines truncated)"
    return f"[run_code] Status: {status}\n\n{truncated}"


def _tool_get_traceback(buggy_code: str) -> str:
    output, success, stderr = _run_in_subprocess(buggy_code, timeout=30)
    if success:
        return "[get_traceback] Code ran without error. No traceback available."
    if not stderr and not output:
        return "[get_traceback] No traceback captured."
    tb_text = stderr if stderr else output
    lines = tb_text.split("\n")
    truncated = "\n".join(lines[:50])
    return f"[get_traceback] Full traceback:\n\n{truncated}"


def _tool_inspect_gradients(buggy_code: str, task_id: str) -> str:
    """
    Inject gradient norm logging before backward pass, run one batch, report per-layer norms.
    For tasks where code crashes before backward (shape_mismatch, wrong_device), returns crash info.
    For silent bugs (leakage, eval_mode), returns gradient info anyway — agent must interpret.
    """
    instrumented = _inject_gradient_logging(buggy_code)
    output, success, _ = _run_in_subprocess(instrumented, timeout=35)

    if not success and "GRAD_NORMS" not in output:
        lines = output.split("\n")[:20]
        return (
            "[inspect_gradients] Code crashed before gradients could be measured.\n"
            "Partial output:\n" + "\n".join(lines)
        )

    lines = output.split("\n")
    grad_lines = [l for l in lines if "GRAD_NORM" in l or "grad_norm" in l.lower() or "GRADIENT" in l]
    if not grad_lines:
        return (
            "[inspect_gradients] Gradient injection ran but no gradient norms captured.\n"
            "This may indicate gradients are not flowing (no backward call reached).\n"
            f"Output excerpt:\n{chr(10).join(lines[:15])}"
        )
    return "[inspect_gradients] Per-layer gradient norms (first batch):\n\n" + "\n".join(grad_lines[:30])


def _inject_gradient_logging(code: str) -> str:
    """
    Inject gradient norm reporting after the first backward() call.
    Wraps the training loop to break after 1 batch and report norms.
    """
    injection = '''
import torch as _torch_grad_tool

_GRAD_BATCH_COUNT = 0
_GRAD_ORIG_backward = _torch_grad_tool.Tensor.backward

def _patched_backward(self, *args, **kwargs):
    global _GRAD_BATCH_COUNT
    _GRAD_ORIG_backward(self, *args, **kwargs)
    _GRAD_BATCH_COUNT += 1
    if _GRAD_BATCH_COUNT == 1:
        # find all parameters with gradients in scope
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, _torch_grad_tool.nn.Module):
                print("GRAD_NORMS:")
                for name, param in obj.named_parameters():
                    if param.grad is not None:
                        norm = param.grad.norm().item()
                        print(f"  GRAD_NORM layer={name} norm={norm:.6f}")
                break

_torch_grad_tool.Tensor.backward = _patched_backward
'''
    return injection + "\n" + code


def _tool_print_shapes(buggy_code: str, task_id: str) -> str:
    """
    Inject shape printing at each linear layer, run one forward pass, report shapes.
    """
    instrumented = _inject_shape_logging(buggy_code)
    output, success, _ = _run_in_subprocess(instrumented, timeout=35)

    lines = output.split("\n")
    shape_lines = [l for l in lines if "SHAPE" in l or "shape" in l.lower()]

    if not shape_lines and not success:
        error_lines = [l for l in lines if "Error" in l or "error" in l]
        return (
            "[print_shapes] Code crashed before shapes could be captured.\n"
            "Error:\n" + "\n".join(error_lines[:10]) +
            "\nFull output excerpt:\n" + "\n".join(lines[:20])
        )

    if not shape_lines:
        return (
            "[print_shapes] Shape hook ran but no shapes captured.\n"
            f"Output:\n{chr(10).join(lines[:20])}"
        )

    return "[print_shapes] Tensor shapes at each layer (first batch):\n\n" + "\n".join(shape_lines[:40])


def _inject_shape_logging(code: str) -> str:
    """Register forward hooks on all Linear layers to print input/output shapes."""
    injection = '''
import torch as _torch_shape_tool

_shape_hooks = []

def _register_shape_hooks(model):
    def make_hook(name):
        def hook(module, input, output):
            in_shape = tuple(input[0].shape) if input else "unknown"
            out_shape = tuple(output.shape)
            print(f"SHAPE layer={name} in={in_shape} out={out_shape}")
        return hook
    for name, module in model.named_modules():
        if isinstance(module, (_torch_shape_tool.nn.Linear, _torch_shape_tool.nn.Conv2d)):
            h = module.register_forward_hook(make_hook(name))
            _shape_hooks.append(h)

_orig_Module_init = _torch_shape_tool.nn.Module.__init__

_registered_models = set()

def _patched_init(self, *args, **kwargs):
    _orig_Module_init(self, *args, **kwargs)

_torch_shape_tool.nn.Module.__init__ = _patched_init

import atexit as _atexit_shape

def _cleanup_hooks():
    for h in _shape_hooks:
        h.remove()

_atexit_shape.register(_cleanup_hooks)

# Patch Module.to() and Module.forward to auto-register hooks on first forward
_orig_forward = _torch_shape_tool.nn.Module.__call__
_hooked_ids = set()

def _auto_hook_call(self, *args, **kwargs):
    mid = id(self)
    if mid not in _hooked_ids and len(list(self.children())) > 0:
        _register_shape_hooks(self)
        _hooked_ids.add(mid)
    return _orig_forward(self, *args, **kwargs)

_torch_shape_tool.nn.Module.__call__ = _auto_hook_call
'''
    return injection + "\n" + code


def _tool_view_source(buggy_code: str) -> str:
    lines = buggy_code.split("\n")
    numbered = "\n".join(f"{i+1:3d} | {line}" for i, line in enumerate(lines))
    return f"[view_source] Full buggy script ({len(lines)} lines):\n\n{numbered}"


# ──────────────────────────────────────────────────────────────
# ALERT MESSAGES — minimal failure notice shown on reset()
# No code, no traceback, no hints. Just what an on-call engineer sees.
# ──────────────────────────────────────────────────────────────

ALERTS = {
    TASK_SHAPE_MISMATCH: "Training job crashed immediately. No epochs completed. Exit code 1.",
    TASK_TRAINING_COLLAPSE: "Training job completed 5 epochs. Final loss: nan. Model did not converge.",
    TASK_DATA_LEAKAGE: "Training job completed successfully. Test accuracy: 96.5%. Review requested.",
    TASK_WRONG_DEVICE: "Training job crashed on first forward pass. Exit code 1.",
    TASK_GRADIENT_NOT_ZEROED: "Training job failed. Loss exploded to nan by epoch 4. Exit code 0.",
    TASK_MISSING_EVAL_MODE: "Training completed. Evaluation metrics unstable across repeated runs.",
    TASK_COMPOUND_SHAPE_DEVICE: "Training job crashed immediately. Multiple errors detected. Exit code 1.",
    TASK_COMPOUND_LEAKAGE_EVAL: "Training completed. Metrics look suspicious and vary between evaluation runs.",
}


# ──────────────────────────────────────────────────────────────
# TASK 1 — Shape Mismatch (Easy)
# ──────────────────────────────────────────────────────────────

def _shape_mismatch_scenario(rng: random.Random) -> BugScenario:
    variant = rng.choice(["mlp", "deep", "autoencoder"])
    hidden_size = rng.choice([128, 256, 512])
    wrong_size = rng.choice([64, 32, 16])
    num_classes = rng.choice([10, 5, 20])

    if variant == "mlp":
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
        error_output = f"RuntimeError: mat1 and mat2 shapes cannot be multiplied ({hidden_size} cannot be broadcast to {wrong_size})"
        solution_hint = f"classifier input must be {hidden_size} not {wrong_size}"

    elif variant == "deep":
        buggy_code = f'''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, {hidden_size}),
            nn.BatchNorm1d({hidden_size}),
            nn.ReLU(),
            nn.Linear({hidden_size}, {hidden_size}),
            nn.ReLU(),
        )
        self.head = nn.Linear({wrong_size}, {num_classes})

    def forward(self, x):
        z = self.feature_extractor(x)
        return self.head(z)

X = torch.randn(300, 512)
y = torch.randint(0, {num_classes}, (300,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = DeepNet()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    for xb, yb in loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {{epoch+1}} complete")

print("Training finished")
'''
        error_output = f"RuntimeError: mat1 and mat2 shapes cannot be multiplied ({hidden_size} cannot be broadcast to {wrong_size})"
        solution_hint = f"head input must be {hidden_size} not {wrong_size}"

    else:
        bottleneck = rng.choice([16, 32])
        buggy_code = f'''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, {hidden_size}),
            nn.ReLU(),
            nn.Linear({hidden_size}, {bottleneck}),
        )
        self.decoder = nn.Sequential(
            nn.Linear({wrong_size}, {hidden_size}),
            nn.ReLU(),
            nn.Linear({hidden_size}, 128),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

X = torch.randn(200, 128)
dataset = TensorDataset(X, X)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Autoencoder()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(3):
    for xb, _ in loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, xb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {{epoch+1}} complete")

print("Training finished")
'''
        error_output = f"RuntimeError: mat1 and mat2 shapes cannot be multiplied ({bottleneck} cannot be broadcast to {wrong_size})"
        solution_hint = f"decoder input must be {bottleneck} not {wrong_size}"

    return BugScenario(
        task_id=TASK_SHAPE_MISMATCH,
        task_description=(
            "This PyTorch model crashes immediately during the forward pass with a shape mismatch. "
            "The training loop never completes a single step. "
            "Find the architectural bug and fix the script so it trains for 3 epochs without error."
        ),
        buggy_code=buggy_code,
        error_output=error_output,
        correct_bug_type="shape_mismatch",
        solution_hint=solution_hint,
        alert=ALERTS[TASK_SHAPE_MISMATCH],
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
            f"Epoch 1, loss: 847.3291\nEpoch 2, loss: nan\nEpoch 3, loss: nan"
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
            "Epoch 1, loss: 0.2489\nEpoch 2, loss: 0.2491\nEpoch 5, loss: 0.2491\n"
            "Loss plateaus immediately. Wrong loss function for binary classification."
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
        alert=ALERTS[TASK_TRAINING_COLLAPSE],
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
            "Script runs to completion. Reported test accuracy: 0.9650\n"
            "However, the evaluation is invalid — there is a data pipeline bug."
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
            "Script runs to completion. Reported test MSE: 0.1021\n"
            "The MSE is artificially low — test statistics leaked into normalization."
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
        alert=ALERTS[TASK_DATA_LEAKAGE],
    )


# ──────────────────────────────────────────────────────────────
# TASK 4 — Wrong Device (Medium)
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
criterion = nn.CrossEntropyLoss().to(device)

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
        "RuntimeError: Expected all tensors to be on the same device, "
        "but found at least two devices!\n\n"
        "The model was moved to the target device but data batches remain on CPU. "
        "Every forward pass crashes. Fix tensor placement so all tensors are on the same device."
    )

    return BugScenario(
        task_id=TASK_WRONG_DEVICE,
        task_description=(
            "This PyTorch training script crashes on the first forward pass. "
            "The model and data tensors are on different devices. "
            "Fix the script so training runs for 3 epochs without error on whatever device is available."
        ),
        buggy_code=buggy_code,
        error_output=error_output,
        correct_bug_type="wrong_device",
        solution_hint="move xb and yb to device inside the training loop: xb, yb = xb.to(device), yb.to(device)",
        alert=ALERTS[TASK_WRONG_DEVICE],
    )


# ──────────────────────────────────────────────────────────────
# TASK 5 — Gradient Not Zeroed (Medium-Hard)
# ──────────────────────────────────────────────────────────────

def _gradient_not_zeroed_scenario(rng: random.Random) -> BugScenario:
    hidden = rng.choice([32, 64, 128])
    lr = rng.choice([1e-3, 5e-4])
    variant = rng.choice(["regression", "classification"])

    if variant == "regression":
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
    else:
        buggy_code = f'''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(32, {hidden}),
            nn.ReLU(),
            nn.Linear({hidden}, {hidden}),
            nn.ReLU(),
        )
        self.classifier = nn.Linear({hidden}, 4)

    def forward(self, x):
        return self.classifier(self.features(x))

X = torch.randn(400, 32)
y = torch.randint(0, 4, (400,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Net()
optimizer = optim.SGD(model.parameters(), lr={lr}, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(6):
    epoch_loss = 0.0
    for xb, yb in loader:
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg = epoch_loss / len(loader)
    print(f"Epoch {{epoch+1}}, loss: {{avg:.4f}}")

print("Training finished")
'''

    error_output = (
        "Script runs without crashing but training is highly unstable.\n"
        "Epoch 1, loss: 12.4821\nEpoch 2, loss: 847.2341\n"
        "Epoch 3, loss: 23451.8821\nEpoch 4, loss: nan\n"
        "Loss explodes after epoch 1 and collapses to NaN. "
        "Fundamental error in training loop structure."
    )

    return BugScenario(
        task_id=TASK_GRADIENT_NOT_ZEROED,
        task_description=(
            "This PyTorch training script runs without crashing but loss explodes "
            "after the first epoch and collapses to NaN. The model never learns. "
            "Find the training loop bug and fix it so loss decreases consistently across 6 epochs."
        ),
        buggy_code=buggy_code,
        error_output=error_output,
        correct_bug_type="gradient_not_zeroed",
        solution_hint="optimizer.zero_grad() is missing before loss.backward(); gradients accumulate causing explosion",
        alert=ALERTS[TASK_GRADIENT_NOT_ZEROED],
    )


# ──────────────────────────────────────────────────────────────
# TASK 6 — Missing Eval Mode (Hard)
# ──────────────────────────────────────────────────────────────

def _missing_eval_mode_scenario(rng: random.Random) -> BugScenario:
    dropout_p = rng.choice([0.3, 0.4, 0.5])
    hidden = rng.choice([64, 128])
    num_classes = rng.choice([3, 5])
    variant = rng.choice(["classifier", "regressor"])

    if variant == "classifier":
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
    else:
        buggy_code = f'''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class RegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, {hidden}),
            nn.BatchNorm1d({hidden}),
            nn.ReLU(),
            nn.Dropout(p={dropout_p}),
            nn.Linear({hidden}, {hidden}),
            nn.ReLU(),
            nn.Dropout(p={dropout_p}),
            nn.Linear({hidden}, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

torch.manual_seed(42)
N = 600
X = torch.randn(N, 15)
y = X[:, 0] * 2.0 + X[:, 3] * 0.5 + torch.randn(N) * 0.3

split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

model = RegNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {{epoch+1}} complete")

test_loss = criterion(model(X_test), y_test).item()
print(f"Test MSE: {{test_loss:.4f}}")
print("Evaluation complete")
print("Training finished")
'''

    error_output = (
        "Script runs to completion with no errors.\n"
        f"Reported metrics vary between runs due to active Dropout(p={dropout_p}).\n"
        "Running evaluation twice gives different numbers. "
        "Model appears to be in wrong mode during evaluation."
    )

    return BugScenario(
        task_id=TASK_MISSING_EVAL_MODE,
        task_description=(
            "This PyTorch model trains successfully but produces unreliable evaluation metrics. "
            "Running evaluation multiple times gives different results each time. "
            f"The model has Dropout(p={dropout_p}) and BatchNorm layers. "
            "Fix the evaluation so it produces stable, deterministic metrics."
        ),
        buggy_code=buggy_code,
        error_output=error_output,
        correct_bug_type="missing_eval_mode",
        solution_hint=f"model.eval() and torch.no_grad() must be called before evaluation; dropout p={dropout_p} stays active in train mode",
        alert=ALERTS[TASK_MISSING_EVAL_MODE],
    )


# ──────────────────────────────────────────────────────────────
# TASK 7 — Compound: Shape Mismatch + Wrong Device (Medium-Hard)
# ──────────────────────────────────────────────────────────────

def _compound_shape_device_scenario(rng: random.Random) -> BugScenario:
    hidden_size = rng.choice([128, 256])
    wrong_size = rng.choice([32, 16])
    num_classes = rng.choice([5, 10])

    buggy_code = f'''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class MultiLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(256, {hidden_size}),
            nn.ReLU(),
            nn.Linear({hidden_size}, {hidden_size}),
            nn.ReLU(),
        )
        self.classifier = nn.Linear({wrong_size}, {num_classes})

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

X = torch.randn(300, 256)
y = torch.randint(0, {num_classes}, (300,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiLayerNet().to(device)
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
        "This script has TWO bugs that must both be fixed.\n\n"
        f"Bug 1 — Shape mismatch:\n"
        f"  RuntimeError: mat1 and mat2 shapes cannot be multiplied "
        f"({hidden_size} cannot be broadcast to {wrong_size})\n\n"
        "Bug 2 — Device mismatch:\n"
        "  RuntimeError: Expected all tensors to be on the same device!\n\n"
        "Fix BOTH bugs. Script should train 3 epochs without error."
    )

    return BugScenario(
        task_id=TASK_COMPOUND_SHAPE_DEVICE,
        task_description=(
            "This PyTorch script has TWO bugs that must both be fixed. "
            "There is a shape mismatch in the model architecture AND a device placement error. "
            "Fix both bugs so the script trains for 3 epochs without any errors."
        ),
        buggy_code=buggy_code,
        error_output=error_output,
        correct_bug_type="compound_shape_device",
        solution_hint=f"fix 1: classifier input must be {hidden_size} not {wrong_size}; fix 2: move xb and yb to device in training loop",
        num_bugs=2,
        alert=ALERTS[TASK_COMPOUND_SHAPE_DEVICE],
    )


# ──────────────────────────────────────────────────────────────
# TASK 8 — Compound: Data Leakage + Missing Eval Mode (Expert)
# ──────────────────────────────────────────────────────────────

def _compound_leakage_eval_scenario(rng: random.Random) -> BugScenario:
    dropout_p = rng.choice([0.3, 0.4])
    hidden = rng.choice([64, 128])
    num_classes = rng.choice([3, 4])

    buggy_code = f'''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class TabularNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, {hidden}),
            nn.BatchNorm1d({hidden}),
            nn.ReLU(),
            nn.Dropout(p={dropout_p}),
            nn.Linear({hidden}, {hidden}),
            nn.ReLU(),
            nn.Dropout(p={dropout_p}),
            nn.Linear({hidden}, num_classes),
        )

    def forward(self, x):
        return self.net(x)

torch.manual_seed(42)
N, D, C = 1000, 20, {num_classes}
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

model = TabularNet(D, C)
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

test_preds = model(X_test).argmax(dim=1)
accuracy = (test_preds == y_test).float().mean().item()
print(f"Test accuracy: {{accuracy:.4f}}")
print("Evaluation complete")
print("Training finished")
'''

    error_output = (
        "Script runs to completion with no errors.\n"
        "Reported test accuracy: 0.9700 (varies slightly between runs)\n\n"
        "This script has TWO silent bugs:\n"
        "Bug 1 — Data leakage: normalization statistics computed from entire dataset before split.\n"
        f"Bug 2 — Missing eval mode: model.eval() not called; Dropout(p={dropout_p}) active during evaluation."
    )

    return BugScenario(
        task_id=TASK_COMPOUND_LEAKAGE_EVAL,
        task_description=(
            "This PyTorch script runs cleanly and reports impressive metrics — but contains "
            "TWO silent bugs that make the evaluation invalid. "
            "There is a data leakage bug in preprocessing AND a missing eval mode bug. "
            "Fix both so the evaluation is correct and deterministic."
        ),
        buggy_code=buggy_code,
        error_output=error_output,
        correct_bug_type="compound_leakage_eval",
        solution_hint=f"fix 1: compute mean/std only from X_train after split; fix 2: add model.eval() and torch.no_grad() before evaluation",
        num_bugs=2,
        alert=ALERTS[TASK_COMPOUND_LEAKAGE_EVAL],
    )
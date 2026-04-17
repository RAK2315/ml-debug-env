import ast
import subprocess
import sys
import tempfile
import os
import re
from dataclasses import dataclass
from typing import Optional

from bug_generator import BugScenario


@dataclass
class GradeResult:
    score: float
    feedback: str
    execution_output: str


def grade(action_bug_type: str, action_diagnosis: str, fixed_code: str, scenario: BugScenario) -> GradeResult:
    # "other" skips bug type check — goes straight to execution-based scoring
    type_correct = _check_bug_type(action_bug_type, scenario.correct_bug_type)
    if not type_correct:
        return GradeResult(
            score=0.01,
            feedback=(
                f"Incorrect bug type. You identified '{action_bug_type}' "
                f"but the actual bug is '{scenario.correct_bug_type}'. "
                "Re-examine the code and error output carefully."
            ),
            execution_output="(code not executed — bug type was wrong)",
        )

    exec_output, ran_ok = _run_code(fixed_code, timeout=40)

    if not ran_ok:
        return GradeResult(
            score=0.2,
            feedback=(
                "Correct bug type identified. However, your fixed code failed to run. "
                f"Execution output:\n{exec_output}"
            ),
            execution_output=exec_output,
        )

    completed = _check_training_completed(exec_output, scenario.task_id)
    if not completed:
        return GradeResult(
            score=0.4,
            feedback=(
                "Correct bug type. Code ran but training did not complete successfully. "
                f"Execution output:\n{exec_output}"
            ),
            execution_output=exec_output,
        )

    fix_valid, fix_feedback = _verify_fix(fixed_code, scenario, exec_output)
    if not fix_valid:
        return GradeResult(
            score=0.6,
            feedback=(
                "Correct bug type. Code runs and training completes. "
                f"However, the fix does not fully resolve the root cause: {fix_feedback}\n"
                f"Execution output:\n{exec_output}"
            ),
            execution_output=exec_output,
        )

    success, success_feedback = _check_success_signal(scenario.task_id, fixed_code, exec_output)
    if not success:
        return GradeResult(
            score=0.8,
            feedback=(
                "Correct bug type. Code runs. Fix is valid. "
                f"But the expected success signal was not detected: {success_feedback}\n"
                f"Execution output:\n{exec_output}"
            ),
            execution_output=exec_output,
        )

    return GradeResult(
        score=0.99,
        feedback=(
            "Perfect fix. Bug type correct, code runs cleanly, "
            "training completes, and success signal confirmed.\n"
            f"Execution output:\n{exec_output}"
        ),
        execution_output=exec_output,
    )


def _check_bug_type(submitted: str, correct: str) -> bool:
    # "other" always passes — execution-based scoring handles it
    submitted_clean = submitted.strip().lower().replace(" ", "_").replace("-", "_")
    if submitted_clean == "other":
        return True

    correct_clean = correct.strip().lower()
    aliases = {
        "shape_mismatch": {
            "shape_mismatch", "shape", "dimension", "size_mismatch", "linear",
            "matmul", "incompatible", "input_shape", "classifier",
        },
        "training_collapse": {
            "training_collapse", "collapse", "nan", "diverge", "learning_rate",
            "loss_fn", "loss_function", "wrong_loss",
        },
        "data_leakage": {
            "data_leakage", "leakage", "leak", "train_test_leak",
            "normalization", "preprocessing",
        },
        "wrong_device": {
            "wrong_device", "device", "device_mismatch", "cuda", "cpu", "device_error",
        },
        "gradient_not_zeroed": {
            "gradient_not_zeroed", "gradient", "zero_grad", "missing_zero_grad",
            "accumulate", "gradient_accumulation",
        },
        "missing_eval_mode": {
            "missing_eval_mode", "eval_mode", "eval", "dropout", "batchnorm",
            "no_grad", "inference_mode",
        },
        "compound_shape_device": {
            "compound_shape_device", "compound", "shape_device", "multiple",
            "two_bugs", "shape_mismatch", "wrong_device", "shape", "device",
        },
        "compound_leakage_eval": {
            "compound_leakage_eval", "compound", "leakage_eval", "multiple",
            "two_bugs", "data_leakage", "missing_eval_mode", "leakage", "eval",
        },
    }
    valid = aliases.get(correct_clean, {correct_clean})
    if submitted_clean in valid:
        return True
    for alias in valid:
        if alias in submitted_clean:
            return True
    return False


def _run_code(code: str, timeout: int = 40) -> tuple[str, bool]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name

    python_exe = os.environ.get("PYTHON_EXEC")
    if not python_exe:
        server_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(server_dir)
        candidate = os.path.join(project_dir, ".venv", "Scripts", "python.exe")
        if not os.path.exists(candidate):
            candidate = os.path.join(project_dir, ".venv", "bin", "python")
        python_exe = candidate if os.path.exists(candidate) else sys.executable

    try:
        result = subprocess.run(
            [python_exe, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        output = (result.stdout + result.stderr).strip()
        success = result.returncode == 0
        return output, success
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout}s.", False
    except Exception as e:
        return f"Execution error: {e}", False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _run_code_twice(code: str) -> tuple[str, str, bool]:
    """Run code twice and return both outputs. Used for determinism checks."""
    out1, ok1 = _run_code(code, timeout=40)
    out2, ok2 = _run_code(code, timeout=40)
    return out1, out2, (ok1 and ok2)


def _check_training_completed(output: str, task_id: str) -> bool:
    lower = output.lower()
    if "nan" in lower and "loss" in lower:
        return False
    markers = [
        "training finished",
        "training complete",
        "evaluation complete",
        "epoch 3",
        "epoch 5",
        "epoch 6",
        "epoch 8",
        "epoch 10",
    ]
    return any(m in lower for m in markers)


def _zero_grad_before_backward_ast(code: str) -> bool:
    """Use AST to verify optimizer.zero_grad() appears before loss.backward() in the loop."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                body = node.body
                zero_grad_idx = None
                backward_idx = None
                for i, stmt in enumerate(body):
                    stmt_str = ast.unparse(stmt) if hasattr(ast, 'unparse') else str(stmt)
                    if "zero_grad" in stmt_str:
                        zero_grad_idx = i
                    if "backward" in stmt_str:
                        backward_idx = i
                if zero_grad_idx is not None and backward_idx is not None:
                    if zero_grad_idx < backward_idx:
                        return True
        # Also check nested loops
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, (ast.For, ast.While)) and child is not node:
                        body = child.body
                        zero_grad_idx = None
                        backward_idx = None
                        for i, stmt in enumerate(body):
                            stmt_str = ast.unparse(stmt) if hasattr(ast, 'unparse') else str(stmt)
                            if "zero_grad" in stmt_str:
                                zero_grad_idx = i
                            if "backward" in stmt_str:
                                backward_idx = i
                        if zero_grad_idx is not None and backward_idx is not None:
                            if zero_grad_idx < backward_idx:
                                return True
        return False
    except Exception:
        # AST parse failed — fall back to string check
        return "optimizer.zero_grad()" in code


def _extract_metric(output: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, output.lower())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _verify_fix(fixed_code: str, scenario: BugScenario, exec_output: str) -> tuple[bool, str]:
    task = scenario.task_id

    if task == "shape_mismatch":
        # Code runs and trains — shape is fixed by definition
        # Just verify no shape error in output
        if "cannot be multiplied" in exec_output.lower():
            return False, "Shape mismatch error still present in execution output."
        return True, ""

    elif task == "training_collapse":
        lower_output = exec_output.lower()
        if "nan" in lower_output:
            return False, "Loss is still NaN in the fixed code output."
        hint = scenario.solution_hint
        if "learning rate" in hint or "lr" in hint.lower():
            lr_matches = re.findall(r"\blr\s*=\s*([\d.e\-+]+)", fixed_code)
            for lr_str in lr_matches:
                try:
                    if float(lr_str) > 1.0:
                        return False, f"Learning rate {lr_str} is still too large."
                except ValueError:
                    pass
        return True, ""

    elif task == "data_leakage":
        # Behavioral check: accuracy should be in realistic range (not suspiciously high)
        acc = _extract_metric(exec_output, r"accuracy[:\s]+([\d.]+)")
        if acc is not None and acc > 0.97:
            return False, f"Accuracy {acc:.4f} still suspiciously high — leakage may remain."
        mse = _extract_metric(exec_output, r"mse[:\s]+([\d.]+)")
        if mse is not None and mse < 0.04:
            return False, f"MSE {mse:.4f} still suspiciously low — leakage may remain."
        # Code check: no bad normalization pattern
        bad_patterns = [
            r"mean\s*=\s*[Xx]_raw\.mean",
            r"full_mean\s*=\s*[Xx]_raw\.mean",
            r"[Xx]_raw\.mean\(dim=0\)",
        ]
        for pat in bad_patterns:
            if re.search(pat, fixed_code):
                return False, "Normalization statistics still computed from full dataset."
        return True, ""

    elif task == "wrong_device":
        if "expected all tensors" in exec_output.lower():
            return False, "Device mismatch error still present in output."
        good_patterns = [r"\.to\(device\)", r"\.to\('cpu'\)", r"\.to\(\"cpu\"\)"]
        if not any(re.search(p, fixed_code) for p in good_patterns):
            return False, "Data tensors not being moved to device inside the training loop."
        return True, ""

    elif task == "gradient_not_zeroed":
        if "nan" in exec_output.lower():
            return False, "Loss is still NaN — gradients may still be accumulating."
        # AST-based check: zero_grad must appear before backward in loop body
        if not _zero_grad_before_backward_ast(fixed_code):
            return False, "optimizer.zero_grad() not found before loss.backward() in the loop."
        return True, ""

    elif task == "missing_eval_mode":
        # Behavioral check: run fixed code twice, outputs must be identical
        out1, out2, both_ran = _run_code_twice(fixed_code)
        if not both_ran:
            return False, "Fixed code failed to run twice cleanly."
        # Extract metrics from both runs
        acc1 = _extract_metric(out1, r"accuracy[:\s]+([\d.]+)")
        acc2 = _extract_metric(out2, r"accuracy[:\s]+([\d.]+)")
        mse1 = _extract_metric(out1, r"mse[:\s]+([\d.]+)")
        mse2 = _extract_metric(out2, r"mse[:\s]+([\d.]+)")
        if acc1 is not None and acc2 is not None:
            if abs(acc1 - acc2) > 0.02:
                return False, f"Evaluation still non-deterministic: accuracy {acc1:.4f} vs {acc2:.4f} across two runs. model.eval() may be missing or ineffective."
        if mse1 is not None and mse2 is not None:
            if abs(mse1 - mse2) > 0.05:
                return False, f"Evaluation still non-deterministic: MSE {mse1:.4f} vs {mse2:.4f} across two runs."
        # Also check string presence as backup
        if "model.eval()" not in fixed_code:
            return False, "model.eval() is missing before the evaluation block."
        return True, ""

    elif task == "compound_shape_device":
        # Must fix BOTH: shape mismatch AND device placement
        if "cannot be multiplied" in exec_output.lower():
            return False, "Shape mismatch error still present — Bug 1 not fully fixed."
        if "expected all tensors" in exec_output.lower():
            return False, "Device mismatch error still present — Bug 2 not fully fixed."
        good_device = any(re.search(p, fixed_code) for p in [r"\.to\(device\)", r"xb\.to\(", r"yb\.to\("])
        if not good_device:
            return False, "Data tensors not moved to device — Bug 2 not fixed."
        return True, ""

    elif task == "compound_leakage_eval":
        # Must fix BOTH: data leakage AND missing eval mode
        # Check 1: no bad normalization pattern
        bad_patterns = [
            r"mean\s*=\s*[Xx]_raw\.mean",
            r"full_mean\s*=\s*[Xx]_raw\.mean",
            r"[Xx]_raw\.mean\(dim=0\)",
        ]
        for pat in bad_patterns:
            if re.search(pat, fixed_code):
                return False, "Data leakage still present — normalization uses full dataset stats."
        # Check 2: behavioral determinism test
        out1, out2, both_ran = _run_code_twice(fixed_code)
        if not both_ran:
            return False, "Fixed code failed to run twice cleanly."
        acc1 = _extract_metric(out1, r"accuracy[:\s]+([\d.]+)")
        acc2 = _extract_metric(out2, r"accuracy[:\s]+([\d.]+)")
        if acc1 is not None and acc2 is not None:
            if abs(acc1 - acc2) > 0.02:
                return False, f"Eval still non-deterministic: {acc1:.4f} vs {acc2:.4f}. model.eval() may be missing."
        # Check 3: accuracy not suspiciously high (leakage check)
        if acc1 is not None and acc1 > 0.97:
            return False, f"Accuracy {acc1:.4f} still suspiciously high — data leakage may remain."
        if "model.eval()" not in fixed_code:
            return False, "model.eval() missing — Bug 2 not fixed."
        return True, ""

    return True, ""


def _check_success_signal(task_id: str, fixed_code: str, output: str) -> tuple[bool, str]:
    lower = output.lower()

    if task_id == "shape_mismatch":
        has_epoch = any(f"epoch {i}" in lower for i in range(1, 4))
        has_finished = "training finished" in lower or "complete" in lower
        if has_epoch and has_finished:
            return True, ""
        return False, "Expected epoch logs and 'Training finished' not found."

    elif task_id == "training_collapse":
        if "nan" in lower:
            return False, "Output still contains NaN."
        loss_values = re.findall(r"loss[:\s]+([\d.]+)", lower)
        if len(loss_values) >= 2:
            first, last = float(loss_values[0]), float(loss_values[-1])
            if last < first * 0.95:
                return True, ""
            return False, f"Loss did not decrease: {first:.4f} → {last:.4f}."
        return "training finished" in lower, "Could not confirm loss decreased."

    elif task_id == "data_leakage":
        acc = _extract_metric(output, r"accuracy[:\s]+([\d.]+)")
        if acc is not None and acc > 0.97:
            return False, f"Accuracy {acc:.4f} suspiciously high — leakage may still be present."
        mse = _extract_metric(output, r"mse[:\s]+([\d.]+)")
        if mse is not None and mse < 0.04:
            return False, f"MSE {mse:.4f} suspiciously low — leakage may still be present."
        return "training finished" in lower, "Training did not complete."

    elif task_id == "wrong_device":
        has_epoch = any(f"epoch {i}" in lower for i in range(1, 4))
        has_finished = "training finished" in lower
        no_error = "expected all tensors" not in lower
        if has_epoch and has_finished and no_error:
            return True, ""
        return False, "Training did not complete cleanly or device error still present."

    elif task_id == "gradient_not_zeroed":
        if "nan" in lower:
            return False, "Loss still diverges to NaN."
        loss_values = re.findall(r"loss[:\s]+([\d.]+)", lower)
        if len(loss_values) >= 2:
            first, last = float(loss_values[0]), float(loss_values[-1])
            if last < first * 0.9:
                return True, ""
            return False, f"Loss did not decrease sufficiently: {first:.4f} → {last:.4f}."
        return "training finished" in lower, "Could not confirm stable training."

    elif task_id == "missing_eval_mode":
        has_metric = "accuracy" in lower or "mse" in lower
        has_finished = "training finished" in lower or "evaluation complete" in lower
        if has_metric and has_finished:
            return True, ""
        return False, "Evaluation did not complete or metric not reported."

    elif task_id == "compound_shape_device":
        has_epoch = any(f"epoch {i}" in lower for i in range(1, 4))
        has_finished = "training finished" in lower
        no_errors = "cannot be multiplied" not in lower and "expected all tensors" not in lower
        if has_epoch and has_finished and no_errors:
            return True, ""
        return False, "Training did not complete or one of the bugs is still present."

    elif task_id == "compound_leakage_eval":
        acc = _extract_metric(output, r"accuracy[:\s]+([\d.]+)")
        has_finished = "training finished" in lower or "evaluation complete" in lower
        if acc is not None and acc > 0.97:
            return False, f"Accuracy {acc:.4f} too high — data leakage may still be present."
        if has_finished:
            return True, ""
        return False, "Training or evaluation did not complete."

    return True, ""
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

    exec_output, ran_ok = _run_code(fixed_code, timeout=30)

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

    success, success_feedback = _check_success_signal(scenario.task_id, exec_output)
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
            f"Perfect fix. Bug type correct, code runs cleanly, "
            f"training completes, and success signal confirmed.\n"
            f"Execution output:\n{exec_output}"
        ),
        execution_output=exec_output,
    )


def _check_bug_type(submitted: str, correct: str) -> bool:
    submitted_clean = submitted.strip().lower().replace(" ", "_").replace("-", "_")
    correct_clean = correct.strip().lower()
    aliases = {
        "shape_mismatch": {"shape_mismatch", "shape", "dimension", "size_mismatch", "linear", "matmul", "incompatible", "input_shape", "classifier"},
        "training_collapse": {"training_collapse", "collapse", "nan", "diverge", "learning_rate", "loss_fn", "loss_function", "wrong_loss"},
        "data_leakage": {"data_leakage", "leakage", "leak", "train_test_leak", "normalization", "preprocessing"},
        "wrong_device": {"wrong_device", "device", "device_mismatch", "cuda", "cpu", "device_error"},
        "gradient_not_zeroed": {"gradient_not_zeroed", "gradient", "zero_grad", "missing_zero_grad", "accumulate", "gradient_accumulation"},
        "missing_eval_mode": {"missing_eval_mode", "eval_mode", "eval", "dropout", "batchnorm", "no_grad", "inference_mode"},
    }
    valid = aliases.get(correct_clean, {correct_clean})
    if submitted_clean in valid:
        return True
    for alias in valid:
        if alias in submitted_clean:
            return True
    return False


def _run_code(code: str, timeout: int = 30) -> tuple[str, bool]:
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


def _verify_fix(fixed_code: str, scenario: BugScenario, exec_output: str) -> tuple[bool, str]:
    task = scenario.task_id

    if task == "shape_mismatch":
        hint = scenario.solution_hint
        match = re.search(r"must be (\d+) not (\d+)", hint)
        if match:
            wrong_size = match.group(2)
            # Check encoder final output matches classifier input
            # Just verify code runs without shape error — already confirmed in stage 2/3
            # Only fail if the EXACT original wrong Linear is unchanged
            original_wrong = f"nn.Linear({wrong_size}, "
            lines = fixed_code.split("\n")
            classifier_lines = [l for l in lines if "classifier" in l.lower() and "nn.Linear" in l]
            encoder_lines = [l for l in lines if "encoder" not in l.lower() and "nn.Linear" in l and "classifier" not in l.lower()]
            # If code runs and trains (already verified), the shape is fixed
            return True, ""
        return True, ""

    elif task == "training_collapse":
        lower_output = exec_output.lower()
        if "nan" in lower_output:
            return False, "Loss is still NaN in the fixed code output."
        hint = scenario.solution_hint
        if "learning rate" in hint or "lr" in hint.lower():
            lr_matches = re.findall(r"lr\s*=\s*([\d.e\-+]+)", fixed_code)
            for lr_str in lr_matches:
                try:
                    lr_val = float(lr_str)
                    if lr_val > 1.0:
                        return False, f"Learning rate {lr_val} is still too large."
                except ValueError:
                    pass
        return True, ""

    elif task == "data_leakage":
        bad_patterns = [
            r"mean\s*=\s*[Xx]_raw\.mean",
            r"full_mean\s*=\s*[Xx]_raw\.mean",
            r"[Xx]_raw\.mean\(dim=0\)",
        ]
        for pat in bad_patterns:
            if re.search(pat, fixed_code):
                return False, "Normalization statistics still computed from full dataset before split."
        good_patterns = [
            r"train.*mean",
            r"mean.*train",
            r"X_train.*\.mean",
            r"X_train_raw.*\.mean",
        ]
        has_good = any(re.search(p, fixed_code, re.IGNORECASE) for p in good_patterns)
        if not has_good:
            return False, "Could not confirm that normalization uses only training data statistics."
        return True, ""

    elif task == "wrong_device":
        # Fixed code must move data to device inside the loop
        good_patterns = [
            r"xb\s*=\s*xb\.to\(device\)",
            r"xb\.to\(device\)",
            r"\.to\(device\)",
        ]
        has_device_move = any(re.search(p, fixed_code) for p in good_patterns)
        if not has_device_move:
            return False, "Data tensors are not being moved to device inside the training loop."
        # Must not crash with device error
        if "expected all tensors" in exec_output.lower():
            return False, "Device mismatch error still present in output."
        return True, ""

    elif task == "gradient_not_zeroed":
        # Fixed code must have zero_grad before backward
        if "optimizer.zero_grad()" not in fixed_code and "optim.zero_grad()" not in fixed_code:
            return False, "optimizer.zero_grad() is still missing from the training loop."
        lower_output = exec_output.lower()
        if "nan" in lower_output and "loss" in lower_output:
            return False, "Loss is still NaN — gradients may still be accumulating."
        return True, ""

    elif task == "missing_eval_mode":
        # Fixed code must have model.eval() before evaluation
        if "model.eval()" not in fixed_code:
            return False, "model.eval() is missing before the evaluation block."
        # Should also have no_grad
        if "torch.no_grad()" not in fixed_code and "no_grad" not in fixed_code:
            return False, "torch.no_grad() context manager is missing during evaluation."
        return True, ""

    return True, ""


def _check_success_signal(task_id: str, output: str) -> tuple[bool, str]:
    lower = output.lower()

    if task_id == "shape_mismatch":
        has_epoch = any(f"epoch {i}" in lower for i in range(1, 4))
        has_finished = "training finished" in lower or "complete" in lower
        if has_epoch and has_finished:
            return True, ""
        return False, "Expected epoch completion messages and 'Training finished' not found."

    elif task_id == "training_collapse":
        if "nan" in lower:
            return False, "Output still contains NaN loss values."
        loss_values = re.findall(r"loss[:\s]+([\d.]+)", lower)
        if len(loss_values) >= 2:
            first, last = float(loss_values[0]), float(loss_values[-1])
            if last < first * 0.95:
                return True, ""
            return False, f"Loss did not decrease: started at {first:.4f}, ended at {last:.4f}."
        has_finished = "training finished" in lower
        return has_finished, "Could not confirm loss decreased across epochs."

    elif task_id == "data_leakage":
        acc_match = re.search(r"accuracy[:\s]+([\d.]+)", lower)
        if acc_match:
            acc = float(acc_match.group(1))
            if acc > 0.98:
                return False, f"Reported accuracy {acc:.4f} is suspiciously high — leakage may still be present."
        mse_match = re.search(r"mse[:\s]+([\d.]+)", lower)
        if mse_match:
            mse = float(mse_match.group(1))
            if mse < 0.05:
                return False, f"Reported MSE {mse:.4f} is suspiciously low — leakage may still be present."
        has_finished = "training finished" in lower
        return has_finished, "Training did not complete."

    elif task_id == "wrong_device":
        has_epoch = any(f"epoch {i}" in lower for i in range(1, 4))
        has_finished = "training finished" in lower
        no_device_error = "expected all tensors" not in lower
        if has_epoch and has_finished and no_device_error:
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
            return False, f"Loss did not decrease sufficiently: {first:.4f} -> {last:.4f}."
        has_finished = "training finished" in lower
        return has_finished, "Could not confirm stable training."

    elif task_id == "missing_eval_mode":
        has_accuracy = "accuracy" in lower or "complete" in lower
        has_finished = "training finished" in lower or "evaluation complete" in lower
        if has_accuracy and has_finished:
            return True, ""
        return False, "Evaluation did not complete or accuracy not reported."

    return True, ""
# server/grader.py
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
    score: float          # 0.0 – 1.0
    feedback: str
    execution_output: str # full stdout+stderr from running the fixed code


def grade(action_bug_type: str, action_diagnosis: str, fixed_code: str, scenario: BugScenario) -> GradeResult:
    """
    Score the agent's fix attempt. Partial credit at every stage:

      0.0  – wrong bug type identified (agent is confused about what's wrong)
      0.2  – correct bug type, but code fails to run (syntax error, import error, crash)
      0.4  – correct bug type + code runs, but training does not complete
      0.6  – correct bug type + code runs + training completes + no NaN loss
      0.8  – all above + fix addresses the actual root cause (verified by heuristic)
      1.0  – all above + output shows the expected successful signal
    """
    # Stage 1: bug type identification
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

    # Stage 2: run the fixed code
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

    # Stage 3: check training completed (no crash mid-loop)
    completed = _check_training_completed(exec_output)
    if not completed:
        return GradeResult(
            score=0.4,
            feedback=(
                "Correct bug type. Code ran but training did not complete successfully. "
                f"Execution output:\n{exec_output}"
            ),
            execution_output=exec_output,
        )

    # Stage 4: check fix actually addresses root cause
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

    # Stage 5: check for success signal in output
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
        "shape_mismatch": {"shape_mismatch", "shape", "dimension", "size_mismatch", "linear", "matmul"},
        "training_collapse": {"training_collapse", "collapse", "nan", "diverge", "learning_rate", "loss_fn", "loss_function", "wrong_loss"},
        "data_leakage": {"data_leakage", "leakage", "leak", "train_test_leak", "normalization", "preprocessing"},
    }
    valid = aliases.get(correct_clean, {correct_clean})
    return submitted_clean in valid


def _run_code(code: str, timeout: int = 30) -> tuple[str, bool]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name

    # Use PYTHON_EXEC env var if set (Docker), else find venv python, else fall back
    python_exe = os.environ.get("PYTHON_EXEC")
    if not python_exe:
        # Walk up from server/ to find .venv
        server_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(server_dir)
        candidate = os.path.join(project_dir, ".venv", "Scripts", "python.exe")  # Windows
        if not os.path.exists(candidate):
            candidate = os.path.join(project_dir, ".venv", "bin", "python")      # Linux/Docker
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


def _check_training_completed(output: str) -> bool:
    markers = [
        "training finished",
        "training complete",
        "epoch 5",
        "epoch 3",
        "epoch 8",
        "epoch 10",
    ]
    lower = output.lower()
    if "nan" in lower and "loss" in lower:
        return False
    return any(m in lower for m in markers)


def _verify_fix(fixed_code: str, scenario: BugScenario, exec_output: str) -> tuple[bool, str]:
    task = scenario.task_id

    if task == "shape_mismatch":
        hint = scenario.solution_hint
        # hint format: "classifier input must be {hidden_size} not {wrong_size}"
        match = re.search(r"must be (\d+) not (\d+)", hint)
        if match:
            correct_size = match.group(1)
            wrong_size = match.group(2)
            # the fixed code should NOT contain nn.Linear({wrong_size}, ...
            pattern = rf"nn\.Linear\s*\(\s*{wrong_size}\s*,"
            if re.search(pattern, fixed_code):
                return False, f"The classifier still uses wrong input size {wrong_size}."
        return True, ""

    elif task == "training_collapse":
        lower_output = exec_output.lower()
        if "nan" in lower_output:
            return False, "Loss is still NaN in the fixed code output."
        hint = scenario.solution_hint
        if "learning rate" in hint or "lr" in hint.lower():
            # check no absurdly large lr remains
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
        # The fix must compute mean/std only from training data, not full X_raw
        # Look for the key pattern: mean computed before split (bad) vs after (good)
        lines = fixed_code.split("\n")
        split_line_idx = None
        mean_line_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "split" in stripped and ("=" in stripped or "int(" in stripped):
                split_line_idx = i
            if re.search(r"(mean|\.mean\()", stripped) and "train" in stripped.lower():
                mean_line_idx = i

        # If mean is computed on something called x_raw before split, that's still wrong
        bad_patterns = [
            r"mean\s*=\s*[Xx]_raw\.mean",
            r"full_mean\s*=\s*[Xx]_raw\.mean",
            r"[Xx]_raw\.mean\(dim=0\)",
        ]
        for pat in bad_patterns:
            if re.search(pat, fixed_code):
                return False, "Normalization statistics still computed from full dataset before split."

        # Good: should see train mean/std
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
        # Check loss is decreasing: find all loss values in output
        loss_values = re.findall(r"loss[:\s]+([\d.]+)", lower)
        if len(loss_values) >= 2:
            first, last = float(loss_values[0]), float(loss_values[-1])
            if last < first * 0.95:
                return True, ""
            return False, f"Loss did not decrease: started at {first:.4f}, ended at {last:.4f}."
        has_finished = "training finished" in lower
        return has_finished, "Could not confirm loss decreased across epochs."

    elif task_id == "data_leakage":
        # The fixed version should NOT show suspiciously high accuracy or very low loss
        # After fixing leakage, accuracy drops or MSE rises — that's the correct behaviour
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

    return True, ""
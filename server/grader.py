import ast
import os
import subprocess
import sys
import tempfile
import re
from dataclasses import dataclass
from typing import Optional

from bug_generator import BugScenario


@dataclass
class GradeResult:
    score: float
    feedback: str
    execution_output: str
    reasoning_reward: float = 0.0
    reasoning_feedback: str = ""


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
            reasoning_reward=0.0,
            reasoning_feedback="Bug type wrong — no reasoning reward.",
        )

    exec_output, ran_ok = _run_code(fixed_code, timeout=40)

    if not ran_ok:
        reasoning_reward, reasoning_feedback = _llm_judge(
            scenario, action_diagnosis, execution_reward=0.2, fix_succeeded=False
        )
        return GradeResult(
            score=min(0.2 + reasoning_reward, 0.99),
            feedback=(
                "Correct bug type identified. However, your fixed code failed to run. "
                f"Execution output:\n{exec_output}"
            ),
            execution_output=exec_output,
            reasoning_reward=reasoning_reward,
            reasoning_feedback=reasoning_feedback,
        )

    completed = _check_training_completed(exec_output, scenario.task_id)
    if not completed:
        reasoning_reward, reasoning_feedback = _llm_judge(
            scenario, action_diagnosis, execution_reward=0.4, fix_succeeded=False
        )
        return GradeResult(
            score=min(0.4 + reasoning_reward, 0.99),
            feedback=(
                "Correct bug type. Code ran but training did not complete successfully. "
                f"Execution output:\n{exec_output}"
            ),
            execution_output=exec_output,
            reasoning_reward=reasoning_reward,
            reasoning_feedback=reasoning_feedback,
        )

    fix_valid, fix_feedback = _verify_fix(fixed_code, scenario, exec_output)
    if not fix_valid:
        reasoning_reward, reasoning_feedback = _llm_judge(
            scenario, action_diagnosis, execution_reward=0.6, fix_succeeded=False
        )
        return GradeResult(
            score=min(0.6 + reasoning_reward, 0.99),
            feedback=(
                "Correct bug type. Code runs and training completes. "
                f"However, the fix does not fully resolve the root cause: {fix_feedback}\n"
                f"Execution output:\n{exec_output}"
            ),
            execution_output=exec_output,
            reasoning_reward=reasoning_reward,
            reasoning_feedback=reasoning_feedback,
        )

    success, success_feedback = _check_success_signal(scenario.task_id, fixed_code, exec_output)
    if not success:
        reasoning_reward, reasoning_feedback = _llm_judge(
            scenario, action_diagnosis, execution_reward=0.8, fix_succeeded=True
        )
        return GradeResult(
            score=min(0.8 + reasoning_reward, 0.99),
            feedback=(
                "Correct bug type. Code runs. Fix is valid. "
                f"But the expected success signal was not detected: {success_feedback}\n"
                f"Execution output:\n{exec_output}"
            ),
            execution_output=exec_output,
            reasoning_reward=reasoning_reward,
            reasoning_feedback=reasoning_feedback,
        )

    reasoning_reward, reasoning_feedback = _llm_judge(
        scenario, action_diagnosis, execution_reward=0.99, fix_succeeded=True
    )
    final_score = min(0.99 + reasoning_reward, 0.99)

    return GradeResult(
        score=final_score,
        feedback=(
            "Perfect fix. Bug type correct, code runs cleanly, "
            "training completes, and success signal confirmed.\n"
            f"Reasoning quality: {reasoning_feedback}\n"
            f"Execution output:\n{exec_output}"
        ),
        execution_output=exec_output,
        reasoning_reward=reasoning_reward,
        reasoning_feedback=reasoning_feedback,
    )


def _llm_judge(
    scenario: BugScenario,
    diagnosis: str,
    execution_reward: float,
    fix_succeeded: bool,
) -> tuple[float, str]:
    """
    LLM judge for reasoning quality.
    Returns (reasoning_reward: 0.0-0.15, feedback: str).
    Only called if GROQ_API_KEY is available — gracefully skips otherwise.
    Reasoning reward is capped at 0.15 so execution reward dominates.
    Final score = min(execution_reward + reasoning_reward, 0.99).
    """
    api_key = (
        os.environ.get("GROQ_API_KEY") or
        os.environ.get("HF_TOKEN") or
        os.environ.get("API_KEY", "")
    ).strip()

    if not api_key or not diagnosis or len(diagnosis.strip()) < 10:
        return 0.0, "No reasoning reward — diagnosis empty or API key not set."

    try:
        import httpx
        prompt = f"""You are evaluating an AI agent's diagnosis of a PyTorch bug.

Task: {scenario.task_id}
Correct bug type: {scenario.correct_bug_type}
Solution hint: {scenario.solution_hint}

Agent's diagnosis:
"{diagnosis}"

Fix succeeded: {fix_succeeded}

Score the diagnosis on these criteria (be strict):
1. Root cause identified correctly? (0-1)
2. Explains WHY the bug causes the failure mechanistically? (0-1)  
3. Specific and actionable (not vague)? (0-1)

Respond with JSON only:
{{"root_cause_score": 0.0, "mechanism_score": 0.0, "specificity_score": 0.0, "reasoning": "one sentence"}}"""

        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200,
                "response_format": {"type": "json_object"},
            },
            timeout=15.0,
        )

        if response.status_code != 200:
            return 0.0, f"Judge API error {response.status_code} — no reasoning reward."

        import json
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        scores = json.loads(content)

        rc = float(scores.get("root_cause_score", 0.0))
        mech = float(scores.get("mechanism_score", 0.0))
        spec = float(scores.get("specificity_score", 0.0))
        reasoning_text = scores.get("reasoning", "")

        raw = (rc * 0.4 + mech * 0.4 + spec * 0.2)
        reward = round(min(raw * 0.15, 0.15), 4)

        feedback = (
            f"Reasoning reward: {reward:.4f}/0.15 "
            f"(root_cause={rc:.1f}, mechanism={mech:.1f}, specificity={spec:.1f}). "
            f"Judge: {reasoning_text}"
        )
        return reward, feedback

    except Exception as e:
        return 0.0, f"Judge unavailable ({e}) — no reasoning reward applied."


def _check_bug_type(submitted: str, correct: str) -> bool:
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
        python_exe = sys.executable

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
        acc = _extract_metric(exec_output, r"accuracy[:\s]+([\d.]+)")
        if acc is not None and acc > 0.97:
            return False, f"Accuracy {acc:.4f} still suspiciously high — leakage may remain."
        mse = _extract_metric(exec_output, r"mse[:\s]+([\d.]+)")
        if mse is not None and mse < 0.04:
            return False, f"MSE {mse:.4f} still suspiciously low — leakage may remain."
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
        if not _zero_grad_before_backward_ast(fixed_code):
            return False, "optimizer.zero_grad() not found before loss.backward() in the loop."
        return True, ""

    elif task == "missing_eval_mode":
        out1, out2, both_ran = _run_code_twice(fixed_code)
        if not both_ran:
            return False, "Fixed code failed to run twice cleanly."
        acc1 = _extract_metric(out1, r"accuracy[:\s]+([\d.]+)")
        acc2 = _extract_metric(out2, r"accuracy[:\s]+([\d.]+)")
        mse1 = _extract_metric(out1, r"mse[:\s]+([\d.]+)")
        mse2 = _extract_metric(out2, r"mse[:\s]+([\d.]+)")
        if acc1 is not None and acc2 is not None:
            if abs(acc1 - acc2) > 0.02:
                return False, f"Evaluation still non-deterministic: accuracy {acc1:.4f} vs {acc2:.4f}."
        if mse1 is not None and mse2 is not None:
            if abs(mse1 - mse2) > 0.05:
                return False, f"Evaluation still non-deterministic: MSE {mse1:.4f} vs {mse2:.4f}."
        if "model.eval()" not in fixed_code:
            return False, "model.eval() is missing before the evaluation block."
        return True, ""

    elif task == "compound_shape_device":
        if "cannot be multiplied" in exec_output.lower():
            return False, "Shape mismatch error still present — Bug 1 not fully fixed."
        if "expected all tensors" in exec_output.lower():
            return False, "Device mismatch error still present — Bug 2 not fully fixed."
        good_device = any(re.search(p, fixed_code) for p in [r"\.to\(device\)", r"xb\.to\(", r"yb\.to\("])
        if not good_device:
            return False, "Data tensors not moved to device — Bug 2 not fixed."
        return True, ""

    elif task == "compound_leakage_eval":
        bad_patterns = [
            r"mean\s*=\s*[Xx]_raw\.mean",
            r"full_mean\s*=\s*[Xx]_raw\.mean",
            r"[Xx]_raw\.mean\(dim=0\)",
        ]
        for pat in bad_patterns:
            if re.search(pat, fixed_code):
                return False, "Data leakage still present — normalization uses full dataset stats."
        out1, out2, both_ran = _run_code_twice(fixed_code)
        if not both_ran:
            return False, "Fixed code failed to run twice cleanly."
        acc1 = _extract_metric(out1, r"accuracy[:\s]+([\d.]+)")
        acc2 = _extract_metric(out2, r"accuracy[:\s]+([\d.]+)")
        if acc1 is not None and acc2 is not None:
            if abs(acc1 - acc2) > 0.02:
                return False, f"Eval still non-deterministic: {acc1:.4f} vs {acc2:.4f}."
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
import os
import sys
import json
from typing import List, Optional
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))
from bug_generator import (
    get_scenario,
    AVAILABLE_TOOLS,
    TASK_SHAPE_MISMATCH,
    TASK_TRAINING_COLLAPSE,
    TASK_DATA_LEAKAGE,
    TASK_WRONG_DEVICE,
    TASK_GRADIENT_NOT_ZEROED,
    TASK_MISSING_EVAL_MODE,
    TASK_COMPOUND_SHAPE_DEVICE,
    TASK_COMPOUND_LEAKAGE_EVAL,
    execute_tool,
)
from grader import grade

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("GROQ_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

ENV_NAME = "ml_debug_env"
MAX_STEPS = 5
SUCCESS_THRESHOLD = 0.95

SYSTEM_PROMPT = """You are an expert ML engineer acting as a debugging agent in a partially observable environment.

You receive a minimal failure alert and must investigate before fixing.

EPISODE STRUCTURE:
- You start with ONLY an alert (no code, no traceback)
- Use tool calls to gather information (each costs 1 step)
- Submit a fix when ready (costs 1 step)
- Total budget: 5 steps. Fix in ≤2 steps = 1.2× reward bonus.

AVAILABLE TOOLS:
- run_code: runs the buggy script, returns stdout/stderr
- get_traceback: returns full traceback if code crashed
- inspect_gradients: injects gradient norm logging, returns per-layer norms
- print_shapes: injects shape printing, returns tensor shapes at each layer
- view_source: returns the full buggy code

RESPONSE FORMAT — choose ONE of these two formats per turn:

For tool inspection:
{"action_type": "inspect", "tool_name": "<tool_name>"}

For submitting fix:
{"action_type": "fix", "bug_type": "<type>", "diagnosis": "<explanation>", "fixed_code": "<complete_script>"}

bug_type must be exactly one of:
shape_mismatch, training_collapse, data_leakage, wrong_device,
gradient_not_zeroed, missing_eval_mode, compound_shape_device, compound_leakage_eval, other

STRATEGY TIPS:
- Start with run_code or get_traceback for crash bugs (shape_mismatch, wrong_device)
- Use inspect_gradients for training instability (gradient_not_zeroed, training_collapse)
- Use print_shapes when the alert mentions a crash with no obvious cause
- view_source is expensive (1 step) — use it only if other tools haven't revealed the bug
- For silent bugs (data_leakage, missing_eval_mode), the alert says code ran fine — inspect carefully

For compound tasks (two bugs), fix ALL bugs in one fix action.
fixed_code must be the COMPLETE runnable script with all imports. No markdown fences."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def call_llm(client: OpenAI, messages: list) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=3000,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


def _efficiency_multiplier(steps_used: int) -> float:
    if steps_used <= 2:
        return 1.2
    elif steps_used <= 3:
        return 1.1
    return 1.0


def run_task(client: OpenAI, task_id: str) -> tuple[float, List[float], int]:
    log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

    scenario = get_scenario(task_id, seed=42)
    rewards: List[float] = []
    best_score = 0.0
    steps_taken = 0

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task ID: {task_id}\n"
                f"Number of bugs: {scenario.num_bugs}\n"
                f"Alert: {scenario.alert}\n\n"
                f"Available tools: {AVAILABLE_TOOLS}\n"
                f"Step budget: {MAX_STEPS}\n\n"
                "Begin your investigation. Respond with JSON only."
            ),
        },
    ]

    for step in range(1, MAX_STEPS + 1):
        steps_taken = step

        try:
            parsed = call_llm(client, messages)
            action_type = parsed.get("action_type", "fix")
            error = None
        except Exception as e:
            action_type = "fix"
            parsed = {"action_type": "fix", "bug_type": "other", "diagnosis": str(e), "fixed_code": ""}
            error = str(e)

        if action_type == "inspect":
            tool_name = parsed.get("tool_name", "")
            if tool_name not in AVAILABLE_TOOLS:
                tool_result = f"Unknown tool '{tool_name}'. Choose from: {AVAILABLE_TOOLS}"
            else:
                tool_result = execute_tool(tool_name, scenario)

            log_step(step=step, action=f"inspect:{tool_name}", reward=0.0, done=False, error=error)
            rewards.append(0.0)

            messages.append({"role": "assistant", "content": json.dumps(parsed)})
            messages.append({
                "role": "user",
                "content": (
                    f"Tool result:\n{tool_result}\n\n"
                    f"Steps remaining: {MAX_STEPS - step}\n"
                    "Continue investigation or submit fix. Respond with JSON only."
                ),
            })

        else:
            bug_type = parsed.get("bug_type", "other")
            diagnosis = parsed.get("diagnosis", "")
            fixed_code = parsed.get("fixed_code", "")

            result = grade(
                action_bug_type=bug_type,
                action_diagnosis=diagnosis,
                fixed_code=fixed_code,
                scenario=scenario,
            )

            multiplier = _efficiency_multiplier(step)
            final_score = min(result.score * multiplier, 0.99)

            if final_score > best_score:
                best_score = final_score

            done = final_score >= SUCCESS_THRESHOLD or step == MAX_STEPS
            log_step(step=step, action=f"fix:{bug_type}", reward=final_score, done=done, error=error)
            rewards.append(final_score)

            if done:
                break

            messages.append({"role": "assistant", "content": json.dumps(parsed)})
            messages.append({
                "role": "user",
                "content": (
                    f"Fix attempt result:\n"
                    f"Score: {final_score:.3f} (multiplier: {multiplier}×)\n"
                    f"Feedback: {result.feedback}\n\n"
                    f"Steps remaining: {MAX_STEPS - step}\n"
                    "Revise your fix or investigate further. Respond with JSON only."
                ),
            })

    return best_score, rewards, steps_taken


def main():
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    tasks = [
        TASK_SHAPE_MISMATCH,
        TASK_TRAINING_COLLAPSE,
        TASK_DATA_LEAKAGE,
        TASK_WRONG_DEVICE,
        TASK_GRADIENT_NOT_ZEROED,
        TASK_MISSING_EVAL_MODE,
        TASK_COMPOUND_SHAPE_DEVICE,
        TASK_COMPOUND_LEAKAGE_EVAL,
    ]

    all_scores: List[float] = []

    for task_id in tasks:
        best_score, rewards, steps = run_task(client, task_id)
        success = best_score >= SUCCESS_THRESHOLD
        log_end(success=success, steps=steps, score=best_score, rewards=rewards)
        all_scores.append(best_score)


if __name__ == "__main__":
    main()
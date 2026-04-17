import os
import sys
import json
from typing import List, Optional
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))
from bug_generator import (
    get_scenario,
    TASK_SHAPE_MISMATCH,
    TASK_TRAINING_COLLAPSE,
    TASK_DATA_LEAKAGE,
    TASK_WRONG_DEVICE,
    TASK_GRADIENT_NOT_ZEROED,
    TASK_MISSING_EVAL_MODE,
    TASK_COMPOUND_SHAPE_DEVICE,
    TASK_COMPOUND_LEAKAGE_EVAL,
)
from grader import grade

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("GROQ_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

ENV_NAME = "ml_debug_env"
MAX_STEPS = 3
SUCCESS_THRESHOLD = 0.95

SYSTEM_PROMPT = """You are an expert ML engineer specializing in debugging PyTorch training code.
You must respond with valid JSON in exactly this format:
{"bug_type": "<EXACT value from list>", "diagnosis": "<clear explanation>", "fixed_code": "<complete corrected Python script>"}

bug_type MUST be exactly one of these strings:
- shape_mismatch
- training_collapse
- data_leakage
- wrong_device
- gradient_not_zeroed
- missing_eval_mode
- compound_shape_device
- compound_leakage_eval
- other

For compound tasks (compound_shape_device, compound_leakage_eval): fix ALL bugs described.
Rules:
- fixed_code must be the COMPLETE script with all imports. Runnable as-is.
- No markdown fences inside JSON values.
- No text outside the JSON object.
- If you see grader feedback from a previous attempt, use it to improve your fix."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def call_llm(client: OpenAI, task_description: str, buggy_code: str, error_output: str, feedback: Optional[str] = None) -> dict:
    user_content = f"Task: {task_description}\n\nBroken script:\n```python\n{buggy_code}\n```\n\nFailure observed:\n{error_output}"
    if feedback:
        user_content += f"\n\nGrader feedback from previous attempt:\n{feedback}\n\nUse this feedback to improve your fix."
    user_content += "\n\nRespond with JSON only."

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content.strip())


def run_task(client: OpenAI, task_id: str) -> tuple[float, List[float], int]:
    log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

    scenario = get_scenario(task_id, seed=42)
    rewards: List[float] = []
    steps_taken = 0
    last_feedback: Optional[str] = None
    best_score = 0.0

    for step in range(1, MAX_STEPS + 1):
        try:
            parsed = call_llm(
                client,
                scenario.task_description,
                scenario.buggy_code,
                scenario.error_output,
                feedback=last_feedback,
            )
            bug_type = parsed.get("bug_type", "other")
            diagnosis = parsed.get("diagnosis", "")
            fixed_code = parsed.get("fixed_code", "")
            error = None
        except Exception as e:
            bug_type, diagnosis, fixed_code = "other", str(e), ""
            error = str(e)

        result = grade(
            action_bug_type=bug_type,
            action_diagnosis=diagnosis,
            fixed_code=fixed_code,
            scenario=scenario,
        )

        score = result.score
        done = score >= SUCCESS_THRESHOLD or step == MAX_STEPS

        rewards.append(score)
        steps_taken = step
        if score > best_score:
            best_score = score

        log_step(step=step, action=bug_type, reward=score, done=done, error=error)

        if done:
            break

        last_feedback = result.feedback

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
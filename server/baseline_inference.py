# server/baseline_inference.py
"""
Baseline inference script.
Uses API_BASE_URL and API_KEY environment variables injected by the validator.
Falls back to Groq if those are not set (local dev).

Model: llama-3.3-70b-versatile
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from bug_generator import (
    get_scenario,
    TASK_SHAPE_MISMATCH,
    TASK_TRAINING_COLLAPSE,
    TASK_DATA_LEAKAGE,
)
from grader import grade

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "llama-3.3-70b-versatile"
SEED = 42

SYSTEM_PROMPT = """You are an expert ML engineer specializing in debugging PyTorch training code.

You will be given a broken Python training script and a description of how it fails.
Your job is to:
1. Identify the exact bug type
2. Explain the root cause clearly
3. Return a complete corrected script that fixes the issue

You must respond with valid JSON in exactly this format:
{
  "bug_type": "<one of: shape_mismatch, training_collapse, data_leakage, other>",
  "diagnosis": "<clear explanation of the root cause>",
  "fixed_code": "<complete corrected Python script, runnable as-is>"
}

Rules:
- fixed_code must be the COMPLETE script, not a diff or partial fix
- fixed_code must include all imports
- Do not add markdown code fences inside the JSON string
- Do not add any text outside the JSON object"""


def build_user_prompt(task_description: str, buggy_code: str, error_output: str) -> str:
    return f"""Task: {task_description}

Broken script:
```python
{buggy_code}
```

Failure observed:
{error_output}

Respond with JSON only."""


def call_llm(client: OpenAI, user_prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content.strip()
    return json.loads(content)


def run_single_task(client: OpenAI, task_id: str) -> dict:
    scenario = get_scenario(task_id, seed=SEED)
    user_prompt = build_user_prompt(
        scenario.task_description,
        scenario.buggy_code,
        scenario.error_output,
    )

    try:
        parsed = call_llm(client, user_prompt)
        bug_type = parsed.get("bug_type", "other")
        diagnosis = parsed.get("diagnosis", "")
        fixed_code = parsed.get("fixed_code", "")
    except (json.JSONDecodeError, KeyError) as e:
        return {
            "task_id": task_id,
            "score": 0.0,
            "feedback": f"Model returned invalid JSON: {e}",
            "bug_type_submitted": "parse_error",
            "execution_output": "",
        }

    result = grade(
        action_bug_type=bug_type,
        action_diagnosis=diagnosis,
        fixed_code=fixed_code,
        scenario=scenario,
    )

    return {
        "task_id": task_id,
        "score": result.score,
        "feedback": result.feedback,
        "bug_type_submitted": bug_type,
        "execution_output": result.execution_output,
    }


def run_baseline_on_all_tasks(api_key: str, base_url: str) -> list:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    results = []
    for task_id in [TASK_SHAPE_MISMATCH, TASK_TRAINING_COLLAPSE, TASK_DATA_LEAKAGE]:
        print(f"  Running baseline on: {task_id} ...", flush=True)
        result = run_single_task(client, task_id)
        results.append(result)
        print(f"  Score: {result['score']} | Bug type: {result['bug_type_submitted']}", flush=True)
    return results


if __name__ == "__main__":
    # Use injected proxy creds if available, fall back to Groq for local dev
    api_key  = os.environ.get("API_KEY")  or os.environ.get("GROQ_API_KEY", "")
    base_url = os.environ.get("API_BASE_URL") or GROQ_BASE_URL

    if not api_key:
        print("ERROR: API_KEY (or GROQ_API_KEY) environment variable not set.")
        sys.exit(1)

    print(f"Running baseline with model: {MODEL}")
    print(f"Base URL: {base_url}")
    print(f"Seed: {SEED}\n")

    results = run_baseline_on_all_tasks(api_key, base_url)

    print("\n=== BASELINE RESULTS ===")
    total = 0.0
    for r in results:
        print(f"\nTask: {r['task_id']}")
        print(f"  Score:    {r['score']}")
        print(f"  Bug type: {r['bug_type_submitted']}")
        print(f"  Feedback: {r['feedback'][:120]}...")
        total += r["score"]

    avg = total / len(results)
    print(f"\nAverage score: {avg:.4f}")
    print("========================")
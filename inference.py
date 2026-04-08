import os
import sys
import json
from openai import OpenAI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))
from bug_generator import get_scenario, TASK_SHAPE_MISMATCH, TASK_TRAINING_COLLAPSE, TASK_DATA_LEAKAGE
from grader import grade

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.getenv("API_KEY") or os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN")

SYSTEM_PROMPT = """You are an expert ML engineer specializing in debugging PyTorch training code.
You must respond with valid JSON in exactly this format:
{"bug_type": "<one of: shape_mismatch, training_collapse, data_leakage, other>", "diagnosis": "<clear explanation>", "fixed_code": "<complete corrected Python script>"}
Rules: fixed_code must be the COMPLETE script with all imports. No markdown fences inside JSON. No text outside JSON."""

def call_llm(client, task_description, buggy_code, error_output):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {task_description}\n\nBroken script:\n```python\n{buggy_code}\n```\n\nFailure observed:\n{error_output}\n\nRespond with JSON only."}
        ],
        temperature=0.0,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content.strip())

def main():
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    tasks = [TASK_SHAPE_MISMATCH, TASK_TRAINING_COLLAPSE, TASK_DATA_LEAKAGE]

    for task_id in tasks:
        print(f"[START] task={task_id}", flush=True)
        scenario = get_scenario(task_id, seed=42)
        try:
            parsed = call_llm(client, scenario.task_description, scenario.buggy_code, scenario.error_output)
            bug_type = parsed.get("bug_type", "other")
            diagnosis = parsed.get("diagnosis", "")
            fixed_code = parsed.get("fixed_code", "")
        except Exception as e:
            bug_type, diagnosis, fixed_code = "other", str(e), ""

        result = grade(action_bug_type=bug_type, action_diagnosis=diagnosis, fixed_code=fixed_code, scenario=scenario)
        print(f"[STEP] step=1 reward={result.score:.4f}", flush=True)
        print(f"[END] task={task_id} score={result.score:.4f} steps=1", flush=True)

    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
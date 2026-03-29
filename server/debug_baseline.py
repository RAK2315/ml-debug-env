# debug_baseline.py
# drop in ml_debug_env/server/, run: python debug_baseline.py

import os
import sys
import json
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from bug_generator import get_scenario, TASK_SHAPE_MISMATCH

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "llama-3.3-70b-versatile"

api_key = os.environ.get("GROQ_API_KEY", "")
if not api_key:
    print("Set GROQ_API_KEY first"); sys.exit(1)

client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

scenario = get_scenario(TASK_SHAPE_MISMATCH, seed=42)

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

user_prompt = f"""Task: {scenario.task_description}

Broken script:
```python
{scenario.buggy_code}
```

Failure observed:
{scenario.error_output}

Respond with JSON only."""

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

raw = response.choices[0].message.content.strip()
print("=== RAW RESPONSE (first 500 chars) ===")
print(raw[:500])
print("\n=== PARSING ===")

parsed = json.loads(raw)
print(f"bug_type: {parsed.get('bug_type')}")
print(f"diagnosis: {parsed.get('diagnosis', '')[:100]}")

fixed_code = parsed.get("fixed_code", "")
print(f"\nfixed_code length: {len(fixed_code)} chars")
print(f"fixed_code first 300 chars:\n{fixed_code[:300]}")

print("\n=== RUNNING FIXED CODE ===")
with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
    f.write(fixed_code)
    tmp = f.name

result = subprocess.run(
    [sys.executable, tmp],
    capture_output=True, text=True, timeout=30
)
print(f"Return code: {result.returncode}")
print(f"STDOUT:\n{result.stdout[:300]}")
print(f"STDERR:\n{result.stderr[:300]}")

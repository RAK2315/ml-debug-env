import os
import sys

os.environ['API_BASE_URL'] = 'https://api.groq.com/openai/v1'
# change line 5 in test2.py from hardcoded key to:
API_KEY = os.environ.get("API_KEY", "")
os.environ['MODEL_NAME'] = 'llama-3.3-70b-versatile'
os.environ['PYTHON_EXEC'] = r'C:\Users\rehtr\AppData\Local\Programs\Python\Python310\python.exe'

sys.path.insert(0, 'server')

print("[1] Importing bug_generator...")
try:
    from bug_generator import get_scenario, TASK_SHAPE_MISMATCH
    print("    OK")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[2] Getting scenario...")
try:
    scenario = get_scenario(TASK_SHAPE_MISMATCH, seed=42)
    print("    OK")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[3] Calling Groq via OpenAI client...")
try:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ['API_KEY'],
        base_url=os.environ['API_BASE_URL']
    )
    r = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {"role": "system", "content": "You are an ML debugger. Respond only in JSON with keys: bug_type, diagnosis, fixed_code"},
            {"role": "user", "content": f"Task: {scenario.task_description}\n\nCode:\n{scenario.buggy_code}\n\nError:\n{scenario.error_output}\n\nReturn JSON only."}
        ],
        temperature=0.0,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    print("    OK:", r.choices[0].message.content[:200])
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[4] Importing grader...")
try:
    from grader import grade
    print("    OK")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[5] Running grader...")
try:
    import json
    parsed = json.loads(r.choices[0].message.content)
    result = grade(
        action_bug_type=parsed.get("bug_type", "other"),
        action_diagnosis=parsed.get("diagnosis", ""),
        fixed_code=parsed.get("fixed_code", ""),
        scenario=scenario,
    )
    print(f"    Score: {result.score}")
    print(f"    Feedback: {result.feedback[:200]}")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
print("\n[6] Fixed code from LLM:")
print(parsed.get("fixed_code", "")[:1000])

print("\nAll steps passed.")
# test_sanity.py
# Run from inside ml_debug_env/ folder: python test_sanity.py

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))
sys.path.insert(0, os.path.dirname(__file__))

print("=== Testing bug_generator ===")
from bug_generator import get_scenario, TASK_SHAPE_MISMATCH, TASK_TRAINING_COLLAPSE, TASK_DATA_LEAKAGE

for task in [TASK_SHAPE_MISMATCH, TASK_TRAINING_COLLAPSE, TASK_DATA_LEAKAGE]:
    s = get_scenario(task, seed=42)
    print(f"  {task}: code={len(s.buggy_code)} chars, bug_type={s.correct_bug_type}")

print("\n=== Testing grader (runs actual code in subprocess) ===")
from grader import grade

scenario = get_scenario(TASK_SHAPE_MISMATCH, seed=42)

# bad fix: wrong bug type
r = grade("data_leakage", "wrong diagnosis", scenario.buggy_code, scenario)
print(f"  Wrong bug type -> score={r.score} (expected 0.0)")
assert r.score == 0.0, f"Expected 0.0 got {r.score}"

# still broken code, right bug type
r = grade("shape_mismatch", "correct type but broken code", scenario.buggy_code, scenario)
print(f"  Right type, still broken code -> score={r.score} (expected 0.2)")
assert r.score == 0.2, f"Expected 0.2 got {r.score}"

print("\n=== Testing environment ===")
from ml_debug_env_environment import MlDebugEnvEnvironment
from models import DebugAction

env = MlDebugEnvEnvironment(task_id=TASK_SHAPE_MISMATCH)
obs = env.reset(seed=42)
print(f"  reset() task_id={obs.task_id}, done={obs.done}, reward={obs.reward}")
assert obs.task_id == TASK_SHAPE_MISMATCH
assert obs.done == False

state = env.state
print(f"  state: episode_id={state.episode_id}, step_count={state.step_count}")
assert state.step_count == 0

# step with a dummy action (will score 0.0 - wrong bug type)
action = DebugAction(
    bug_type="other",
    diagnosis="not sure",
    fixed_code=obs.buggy_code,
)
obs2 = env.step(action)
print(f"  step() score={obs2.grader_score}, done={obs2.done}, step_number={obs2.step_number}")
assert obs2.grader_score == 0.0
assert obs2.step_number == 1

print("\n=== ALL TESTS PASSED ===")
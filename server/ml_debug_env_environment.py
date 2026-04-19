import sys
import os
from uuid import uuid4
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import DebugAction, DebugObservation, DebugState
from bug_generator import (
    get_scenario,
    execute_tool,
    BugScenario,
    ALL_TASKS,
    AVAILABLE_TOOLS,
    TASK_SHAPE_MISMATCH,
    TASK_TRAINING_COLLAPSE,
    TASK_DATA_LEAKAGE,
    TASK_WRONG_DEVICE,
    TASK_GRADIENT_NOT_ZEROED,
    TASK_MISSING_EVAL_MODE,
    TASK_COMPOUND_SHAPE_DEVICE,
    TASK_COMPOUND_LEAKAGE_EVAL,
)
from grader import grade, GradeResult
from adversarial_scheduler import AdversarialScheduler

MAX_STEPS = 5
SUCCESS_THRESHOLD = 0.95


def _efficiency_multiplier(steps_used: int, total_steps: int) -> float:
    """
    Reward agents that fix bugs efficiently.
    steps_used = number of steps taken when fix was submitted (1-indexed).
    """
    if steps_used <= 2:
        return 1.2
    elif steps_used <= 3:
        return 1.1
    else:
        return 1.0


class MlDebugEnvEnvironment(Environment):
    """
    ML Debugging Environment — 8 tasks, easy → expert.
    Partially observable: agent sees only a minimal alert on reset().
    Must use tool calls (inspect actions) to gather information before fixing.

    Episode structure:
      - reset() → minimal alert, available tools, step budget
      - step(action_type="inspect", tool_name=X) → tool output (costs 1 step)
      - step(action_type="fix", bug_type=X, ...) → grader score (costs 1 step)
      - Max 5 steps total across all inspect + fix actions

    Efficiency bonus:
      - Fix correct in ≤2 total steps → score × 1.2 (capped at 0.99)
      - Fix correct in ≤3 total steps → score × 1.1
      - Fix in 4-5 steps → score × 1.0

    Single-bug tasks (6):
      shape_mismatch, training_collapse, wrong_device,
      gradient_not_zeroed, data_leakage, missing_eval_mode

    Compound tasks — TWO bugs per script (2):
      compound_shape_device, compound_leakage_eval
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_id: Optional[str] = None):
        super().__init__()
        self._task_id: Optional[str] = task_id
        self._current_scenario: Optional[BugScenario] = None
        self._state = DebugState(
            episode_id=None,
            step_count=0,
            task_id="",
            max_steps=MAX_STEPS,
            current_score=0.0,
            attempts=0,
            tools_used=[],
            fix_submitted=False,
        )
        self._episode_count = 0
        self._scheduler = AdversarialScheduler(ALL_TASKS)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> DebugObservation:
        active_task = task_id or self._task_id or self._scheduler.next_task()
        effective_seed = seed if seed is not None else self._scheduler.next_seed(active_task)
        scenario = get_scenario(active_task, seed=effective_seed)

        self._current_scenario = scenario
        self._state = DebugState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=active_task,
            max_steps=MAX_STEPS,
            current_score=0.0,
            attempts=0,
            tools_used=[],
            fix_submitted=False,
        )

        return DebugObservation(
            task_id=active_task,
            alert=scenario.alert,
            available_tools=AVAILABLE_TOOLS,
            step_budget=MAX_STEPS,
            step_number=0,
            num_bugs=scenario.num_bugs,
            action_type=None,
            tool_name=None,
            tool_result=None,
            grader_score=None,
            grader_feedback=None,
            execution_result=None,
            done=False,
            reward=None,
            efficiency_multiplier=None,
        )

    def step(
        self,
        action: DebugAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> DebugObservation:
        if self._current_scenario is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        steps_remaining = MAX_STEPS - self._state.step_count

        if action.action_type == "inspect":
            return self._handle_inspect(action, steps_remaining)
        elif action.action_type == "fix":
            return self._handle_fix(action, steps_remaining)
        else:
            self._state.step_count -= 1
            raise ValueError(f"Unknown action_type: '{action.action_type}'. Must be 'inspect' or 'fix'.")

    def _handle_inspect(self, action: DebugAction, steps_remaining: int) -> DebugObservation:
        tool_name = action.tool_name or ""
        if tool_name not in AVAILABLE_TOOLS:
            tool_result = (
                f"Unknown tool: '{tool_name}'. "
                f"Available tools: {AVAILABLE_TOOLS}"
            )
        else:
            tool_result = execute_tool(tool_name, self._current_scenario)
            self._state.tools_used.append(tool_name)

        done = self._state.step_count >= MAX_STEPS

        return DebugObservation(
            task_id=self._state.task_id,
            alert=self._current_scenario.alert,
            available_tools=AVAILABLE_TOOLS,
            step_budget=steps_remaining,
            step_number=self._state.step_count,
            num_bugs=self._current_scenario.num_bugs,
            action_type="inspect",
            tool_name=tool_name,
            tool_result=tool_result,
            grader_score=None,
            grader_feedback=None,
            execution_result=None,
            done=done,
            reward=0.0,
            efficiency_multiplier=None,
        )

    def _handle_fix(self, action: DebugAction, steps_remaining: int) -> DebugObservation:
        self._state.attempts += 1
        self._state.fix_submitted = True

        bug_type = action.bug_type or "other"
        diagnosis = action.diagnosis or ""
        fixed_code = action.fixed_code or ""

        result: GradeResult = grade(
            action_bug_type=bug_type,
            action_diagnosis=diagnosis,
            fixed_code=fixed_code,
            scenario=self._current_scenario,
        )

        multiplier = _efficiency_multiplier(self._state.step_count, MAX_STEPS)
        final_score = min(result.score * multiplier, 0.99)

        if final_score > self._state.current_score:
            self._state.current_score = final_score

        done = final_score >= SUCCESS_THRESHOLD or self._state.step_count >= MAX_STEPS

        if done:
            self._scheduler.record(self._state.task_id, final_score)

        return DebugObservation(
            task_id=self._state.task_id,
            alert=self._current_scenario.alert,
            available_tools=AVAILABLE_TOOLS,
            step_budget=steps_remaining,
            step_number=self._state.step_count,
            num_bugs=self._current_scenario.num_bugs,
            action_type="fix",
            tool_name=None,
            tool_result=None,
            grader_score=final_score,
            grader_feedback=result.feedback,
            execution_result=result.execution_output,
            done=done,
            reward=final_score,
            efficiency_multiplier=multiplier,
        )

    @property
    def state(self) -> DebugState:
        return self._state

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="ML Debugging Environment",
            description=(
                "Partially observable RL environment where agents debug broken PyTorch training scripts. "
                "Agent sees only a minimal failure alert on reset — no code, no traceback. "
                "Must use tool calls (run_code, get_traceback, inspect_gradients, print_shapes, view_source) "
                "to investigate before submitting a fix. "
                "5 steps total per episode. Efficiency bonus: fix in ≤2 steps → ×1.2 reward. "
                "8 tasks: six single-bug (easy→hard), two compound double-bug tasks (expert). "
                "Execution-based grading in subprocess."
            ),
            version="4.0.0",
            author="ml-debug-env",
        )
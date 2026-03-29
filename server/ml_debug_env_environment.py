# server/ml_debug_env_environment.py
import sys
import os
from uuid import uuid4
from typing import Optional

# server/ is the working dir inside Docker; add parent for models import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import DebugAction, DebugObservation, DebugState
from bug_generator import (
    get_scenario,
    get_random_task,
    BugScenario,
    TASK_SHAPE_MISMATCH,
    TASK_TRAINING_COLLAPSE,
    TASK_DATA_LEAKAGE,
)
from grader import grade, GradeResult

TASK_ORDER = [TASK_SHAPE_MISMATCH, TASK_TRAINING_COLLAPSE, TASK_DATA_LEAKAGE]
MAX_STEPS = 3


class MlDebugEnvEnvironment(Environment):
    """
    ML Debugging Environment.

    The agent receives a broken PyTorch training script and must:
      1. Identify the bug type
      2. Explain the root cause
      3. Return a complete corrected script

    Three tasks of increasing difficulty:
      - shape_mismatch    (easy)   : explicit crash, wrong linear layer size
      - training_collapse (medium) : silent failure, NaN loss or wrong loss fn
      - data_leakage      (hard)   : everything looks fine, evaluation is invalid

    Episodes are single-step by default (one fix attempt = one episode).
    MAX_STEPS=3 allows retry attempts with partial credit carried forward.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_id: Optional[str] = None):
        super().__init__()
        self._task_id: Optional[str] = task_id  # None = rotate through tasks
        self._current_scenario: Optional[BugScenario] = None
        self._state = DebugState(
            episode_id=None,
            step_count=0,
            task_id="",
            max_steps=MAX_STEPS,
            current_score=0.0,
            attempts=0,
        )
        self._episode_count = 0  # used to rotate tasks when task_id is None

    # ------------------------------------------------------------------ #
    #  reset()                                                             #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> DebugObservation:
        """
        Start a new episode. Returns the buggy script + error output.

        Args:
            seed:       Random seed for reproducible scenario generation.
            episode_id: Optional custom episode identifier.
            task_id:    Pin to a specific task ('shape_mismatch',
                        'training_collapse', 'data_leakage').
                        If None, rotates through tasks in order.
        """
        active_task = task_id or self._task_id or self._next_task()
        scenario = get_scenario(active_task, seed=seed)

        self._current_scenario = scenario
        self._state = DebugState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=active_task,
            max_steps=MAX_STEPS,
            current_score=0.0,
            attempts=0,
        )

        return DebugObservation(
            task_id=active_task,
            task_description=scenario.task_description,
            buggy_code=scenario.buggy_code,
            error_output=scenario.error_output,
            execution_result=None,
            grader_score=None,
            grader_feedback=None,
            step_number=0,
            done=False,
            reward=None,
        )

    # ------------------------------------------------------------------ #
    #  step()                                                              #
    # ------------------------------------------------------------------ #

    def step(
        self,
        action: DebugAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> DebugObservation:
        """
        Evaluate the agent's fix attempt and return a scored observation.

        The agent must supply:
          - bug_type    : category of the bug
          - diagnosis   : plain-language explanation
          - fixed_code  : complete corrected Python script
        """
        if self._current_scenario is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        self._state.attempts += 1

        result: GradeResult = grade(
            action_bug_type=action.bug_type,
            action_diagnosis=action.diagnosis,
            fixed_code=action.fixed_code,
            scenario=self._current_scenario,
        )

        # Keep best score across retry attempts
        if result.score > self._state.current_score:
            self._state.current_score = result.score

        done = (
            result.score == 1.0
            or self._state.step_count >= MAX_STEPS
        )

        return DebugObservation(
            task_id=self._state.task_id,
            task_description=self._current_scenario.task_description,
            buggy_code=self._current_scenario.buggy_code,
            error_output=self._current_scenario.error_output,
            execution_result=result.execution_output,
            grader_score=result.score,
            grader_feedback=result.feedback,
            step_number=self._state.step_count,
            done=done,
            reward=result.score,
        )

    # ------------------------------------------------------------------ #
    #  state property                                                      #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> DebugState:
        return self._state

    # ------------------------------------------------------------------ #
    #  helpers                                                             #
    # ------------------------------------------------------------------ #

    def _next_task(self) -> str:
        """Rotate through tasks so consecutive episodes cover all three."""
        task = TASK_ORDER[self._episode_count % len(TASK_ORDER)]
        self._episode_count += 1
        return task

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="ML Debugging Environment",
            description=(
                "An RL environment where agents debug broken PyTorch training scripts. "
                "Three tasks of increasing difficulty: shape mismatch (easy), "
                "training collapse (medium), and silent data leakage (hard). "
                "Agents receive a buggy script + error output and must return "
                "a corrected script. The grader executes the fix and scores 0.0–1.0 "
                "with partial credit at each stage."
            ),
            version="1.0.0",
            author="ml-debug-env",
        )
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
    get_random_task,
    BugScenario,
    TASK_SHAPE_MISMATCH,
    TASK_TRAINING_COLLAPSE,
    TASK_DATA_LEAKAGE,
    TASK_WRONG_DEVICE,
    TASK_GRADIENT_NOT_ZEROED,
    TASK_MISSING_EVAL_MODE,
    ALL_TASKS,
)
from grader import grade, GradeResult

MAX_STEPS = 3


class MlDebugEnvEnvironment(Environment):
    """
    ML Debugging Environment — 6 tasks, easy → hard.

    Tasks:
      shape_mismatch      (easy)        — explicit crash, wrong linear layer size
      training_collapse   (medium)      — NaN loss or wrong loss function
      data_leakage        (hard)        — silent, evaluation is invalid
      wrong_device        (medium)      — CPU/CUDA tensor mismatch, explicit crash
      gradient_not_zeroed (medium-hard) — missing zero_grad, loss explodes
      missing_eval_mode   (hard)        — no model.eval(), unreliable metrics

    Episodes are single-step by default. MAX_STEPS=3 allows retry with
    grader feedback fed back to the agent.
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
        )
        self._episode_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> DebugObservation:
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

    def step(
        self,
        action: DebugAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> DebugObservation:
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

        if result.score > self._state.current_score:
            self._state.current_score = result.score

        done = (
            result.score >= 0.95
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

    @property
    def state(self) -> DebugState:
        return self._state

    def _next_task(self) -> str:
        task = ALL_TASKS[self._episode_count % len(ALL_TASKS)]
        self._episode_count += 1
        return task

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="ML Debugging Environment",
            description=(
                "An RL environment where agents debug broken PyTorch training scripts. "
                "Six tasks of increasing difficulty: shape mismatch (easy), "
                "training collapse (medium), wrong device (medium), "
                "gradient not zeroed (medium-hard), data leakage (hard), "
                "and missing eval mode (hard). "
                "Agents receive a buggy script and must return a corrected version. "
                "The grader executes the fix and scores 0.01–0.99 with partial credit."
            ),
            version="2.0.0",
            author="ml-debug-env",
        )
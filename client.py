# ml_debug_env/client.py
"""ML Debug Env — EnvClient for connecting to the ML debugging environment server."""

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import DebugAction, DebugObservation, DebugState


class MlDebugEnvClient(EnvClient[DebugAction, DebugObservation, DebugState]):
    """
    Client for the ML Debug Env environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance gets its own dedicated episode session on the server.

    The agent must:
      1. Identify the bug type (shape_mismatch | training_collapse | data_leakage | other)
      2. Explain the root cause (diagnosis)
      3. Return a complete corrected Python script (fixed_code)

    The grader executes the fixed_code in an isolated subprocess and returns
    a score from 0.0 (wrong bug type) to 1.0 (perfect fix confirmed).

    Example (async)::

        async with MlDebugEnvClient(base_url="http://localhost:8000") as client:
            result = await client.reset()
            obs = result.observation
            print(obs.task_id)      # e.g. "shape_mismatch"
            print(obs.buggy_code)   # the broken script
            print(obs.error_output) # what went wrong

            fix = DebugAction(
                bug_type="shape_mismatch",
                diagnosis="classifier Linear input dim is wrong",
                fixed_code="import torch\\n...",
            )
            result = await client.step(fix)
            print(result.observation.grader_score)   # 0.0 – 1.0
            print(result.observation.grader_feedback)

    Example (sync)::

        with MlDebugEnvClient(base_url="http://localhost:8000").sync() as client:
            result = client.reset()
            result = client.step(DebugAction(
                bug_type="training_collapse",
                diagnosis="learning rate too high, causes NaN",
                fixed_code="...",
            ))
            print(result.observation.grader_score)
    """

    def _step_payload(self, action: DebugAction) -> Dict:
        """Convert DebugAction to JSON payload for the /step WebSocket message."""
        return {
            "bug_type": action.bug_type,
            "diagnosis": action.diagnosis,
            "fixed_code": action.fixed_code,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DebugObservation]:
        """Parse server response into StepResult[DebugObservation]."""
        obs_data = payload.get("observation", {})
        observation = DebugObservation(
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            buggy_code=obs_data.get("buggy_code", ""),
            error_output=obs_data.get("error_output", ""),
            execution_result=obs_data.get("execution_result"),
            grader_score=obs_data.get("grader_score"),
            grader_feedback=obs_data.get("grader_feedback"),
            step_number=obs_data.get("step_number", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> DebugState:
        """Parse server response into DebugState."""
        return DebugState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            max_steps=payload.get("max_steps", 3),
            current_score=payload.get("current_score", 0.0),
            attempts=payload.get("attempts", 0),
        )

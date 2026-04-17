from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class DebugAction(Action):
    """
    The agent's response to a broken ML script.
    """
    bug_type: str = Field(
        ...,
        description=(
            "Category of the bug identified. Must be one of: "
            "'shape_mismatch', 'training_collapse', 'data_leakage', "
            "'wrong_device', 'gradient_not_zeroed', 'missing_eval_mode', "
            "'compound_shape_device', 'compound_leakage_eval', 'other'"
        )
    )
    diagnosis: str = Field(
        ...,
        description="Plain-language explanation of the root cause."
    )
    fixed_code: str = Field(
        ...,
        description="The complete corrected Python script. Must be runnable as-is. Include all imports."
    )


class DebugObservation(Observation):
    """
    What the agent sees at each step.
    """
    task_id: str = Field(..., description="Which task is active")
    task_description: str = Field(..., description="Natural language description of what to fix")
    buggy_code: str = Field(..., description="The broken Python script")
    error_output: str = Field(..., description="Traceback or behavioral failure description")
    execution_result: Optional[str] = Field(None, description="stdout+stderr from running the agent's fix")
    grader_score: Optional[float] = Field(None, description="Score 0.01-0.99")
    grader_feedback: Optional[str] = Field(None, description="Explanation of the score")
    step_number: int = Field(0, description="Current step within this episode")
    num_bugs: int = Field(1, description="Number of bugs in this task (1 or 2 for compound tasks)")


class DebugState(State):
    """
    Internal episode metadata.
    """
    task_id: str = Field("", description="Active task identifier")
    max_steps: int = Field(3, description="Maximum steps allowed per episode")
    current_score: float = Field(0.0, description="Best score achieved so far this episode")
    attempts: int = Field(0, description="Number of fix attempts made")
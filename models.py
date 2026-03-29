# models.py
from typing import Optional, Literal
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class DebugAction(Action):
    """
    The agent's response to a broken ML script.
    
    The agent must identify the bug type, explain the root cause,
    and provide a corrected version of the full script.
    """
    bug_type: str = Field(
        ...,
        description=(
            "Category of the bug identified. Must be one of: "
            "'shape_mismatch', 'training_collapse', 'data_leakage', 'other'"
        )
    )
    diagnosis: str = Field(
        ...,
        description=(
            "Plain-language explanation of the root cause. "
            "What is wrong and why it causes the observed failure."
        )
    )
    fixed_code: str = Field(
        ...,
        description=(
            "The complete corrected Python script. Must be runnable as-is. "
            "Do not truncate. Include all imports and all functions."
        )
    )


class DebugObservation(Observation):
    """
    What the agent sees at each step.
    
    On reset: the broken script + error output.
    After step: execution result of the agent's fix + grader score.
    """
    task_id: str = Field(..., description="Which task is active: 'shape_mismatch', 'training_collapse', or 'data_leakage'")
    task_description: str = Field(..., description="Natural language description of what the agent must fix")
    buggy_code: str = Field(..., description="The broken Python script the agent must debug")
    error_output: str = Field(..., description="The stderr/traceback or behavioral failure description seen when running the buggy script")
    execution_result: Optional[str] = Field(None, description="stdout+stderr from running the agent's fixed code (None on reset)")
    grader_score: Optional[float] = Field(None, description="Score 0.0-1.0 from the grader (None on reset, set after step)")
    grader_feedback: Optional[str] = Field(None, description="Human-readable explanation of why the score was assigned")
    step_number: int = Field(0, description="Current step within this episode")


class DebugState(State):
    """
    Internal episode metadata.
    Extends the base State (which has episode_id and step_count).
    """
    task_id: str = Field("", description="Active task identifier")
    max_steps: int = Field(3, description="Maximum steps allowed per episode")
    current_score: float = Field(0.0, description="Best score achieved so far this episode")
    attempts: int = Field(0, description="Number of fix attempts made")
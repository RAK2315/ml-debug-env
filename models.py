from typing import Optional, List, Literal
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


AVAILABLE_TOOLS = ["run_code", "get_traceback", "inspect_gradients", "print_shapes", "view_source"]


class DebugAction(Action):
    """
    Two action types:
      action_type="inspect" — call a tool to gather information (costs 1 step)
      action_type="fix"     — submit a fix attempt (costs 1 step)

    For inspect: set tool_name to one of the available tools.
    For fix: set bug_type, diagnosis, fixed_code.
    """
    action_type: Literal["inspect", "fix"] = Field(
        ...,
        description="'inspect' to use a diagnostic tool, 'fix' to submit a fix."
    )
    tool_name: Optional[str] = Field(
        None,
        description=(
            "Tool to call (only for action_type='inspect'). "
            "One of: run_code, get_traceback, inspect_gradients, print_shapes, view_source"
        )
    )
    bug_type: Optional[str] = Field(
        None,
        description=(
            "Only for action_type='fix'. Category of the bug identified. "
            "One of: shape_mismatch, training_collapse, data_leakage, wrong_device, "
            "gradient_not_zeroed, missing_eval_mode, compound_shape_device, compound_leakage_eval, other"
        )
    )
    diagnosis: Optional[str] = Field(
        None,
        description="Only for action_type='fix'. Plain-language explanation of the root cause."
    )
    fixed_code: Optional[str] = Field(
        None,
        description="Only for action_type='fix'. Complete corrected Python script. Must be runnable as-is."
    )


class DebugObservation(Observation):
    """
    What the agent sees at each step.

    On reset(): minimal alert only — no buggy code, no error output.
    After inspect action: tool_result contains what the tool found.
    After fix action: grader_score and grader_feedback populated.
    """
    task_id: str = Field(..., description="Which task is active")
    alert: str = Field(..., description="Minimal failure alert shown on reset — e.g. 'Training job failed. Final loss: nan.'")
    available_tools: List[str] = Field(default_factory=list, description="Tools the agent can call")
    step_budget: int = Field(5, description="Total steps remaining (inspect + fix combined)")
    step_number: int = Field(0, description="Current step within this episode")
    num_bugs: int = Field(1, description="Number of bugs in this task (1 or 2 for compound tasks)")

    action_type: Optional[str] = Field(None, description="What action was just taken")
    tool_name: Optional[str] = Field(None, description="Which tool was just called (if inspect)")
    tool_result: Optional[str] = Field(None, description="Output from the tool call (if inspect)")

    grader_score: Optional[float] = Field(None, description="Score 0.01-0.99 (only after fix action)")
    grader_feedback: Optional[str] = Field(None, description="Grader explanation (only after fix action)")
    execution_result: Optional[str] = Field(None, description="Raw execution output from fix attempt")

    done: bool = Field(False, description="Whether the episode is over")
    reward: Optional[float] = Field(None, description="Reward for the last action")
    efficiency_multiplier: Optional[float] = Field(None, description="Bonus applied for efficient fix (1.0-1.2)")


class DebugState(State):
    """Internal episode metadata."""
    episode_id: Optional[str] = Field(None, description="Unique episode identifier")
    task_id: str = Field("", description="Active task identifier")
    max_steps: int = Field(5, description="Maximum steps allowed per episode")
    current_score: float = Field(0.0, description="Best score achieved so far this episode")
    attempts: int = Field(0, description="Number of fix attempts made")
    tools_used: List[str] = Field(default_factory=list, description="Tools called this episode")
    fix_submitted: bool = Field(False, description="Whether a fix has been submitted")
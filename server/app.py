# server/app.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import asyncio
from typing import Any, Dict, List


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openenv.core.env_server.http_server import create_app

from models import DebugAction, DebugObservation, DebugState
from ml_debug_env_environment import MlDebugEnvEnvironment
from bug_generator import (
    TASK_SHAPE_MISMATCH,
    TASK_TRAINING_COLLAPSE,
    TASK_DATA_LEAKAGE,
    get_scenario,
)
from grader import grade

# ------------------------------------------------------------------ #
#  Base OpenEnv app (handles /reset /step /state /health /ws /schema) #
# ------------------------------------------------------------------ #

app = create_app(
    MlDebugEnvEnvironment,
    DebugAction,
    DebugObservation,
    env_name="ml_debug_env",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
#  /tasks  — list tasks + action schema                               #
# ------------------------------------------------------------------ #

TASK_DEFINITIONS = [
    {
        "task_id": TASK_SHAPE_MISMATCH,
        "name": "Shape Mismatch",
        "difficulty": "easy",
        "description": (
            "A PyTorch classifier crashes immediately with a RuntimeError during the "
            "forward pass. The error message is explicit in the traceback. "
            "Fix the architectural bug so the model trains for 3 epochs without error."
        ),
        "success_criteria": "Code runs to completion; all epoch logs print; no RuntimeError.",
    },
    {
        "task_id": TASK_TRAINING_COLLAPSE,
        "name": "Training Collapse",
        "difficulty": "medium",
        "description": (
            "A PyTorch training script runs without crashing but the model completely "
            "fails to learn. Loss diverges to NaN or plateaus immediately. "
            "Fix the training bug so loss decreases consistently across all epochs."
        ),
        "success_criteria": "Loss decreases across epochs; no NaN values in output.",
    },
    {
        "task_id": TASK_DATA_LEAKAGE,
        "name": "Silent Data Leakage",
        "difficulty": "hard",
        "description": (
            "A PyTorch training script runs cleanly and reports impressive metrics. "
            "But the evaluation is fundamentally invalid due to a data pipeline mistake. "
            "Find the data leakage bug and fix it so the evaluation reflects true generalisation."
        ),
        "success_criteria": (
            "Normalization statistics computed only from training data; "
            "test set metrics reflect genuine generalisation."
        ),
    },
]

ACTION_SCHEMA = {
    "type": "object",
    "required": ["bug_type", "diagnosis", "fixed_code"],
    "properties": {
        "bug_type": {
            "type": "string",
            "description": "Category of bug identified.",
            "enum": ["shape_mismatch", "training_collapse", "data_leakage", "other"],
        },
        "diagnosis": {
            "type": "string",
            "description": "Plain-language explanation of the root cause.",
        },
        "fixed_code": {
            "type": "string",
            "description": (
                "The complete corrected Python script. Must be runnable as-is. "
                "Include all imports. Do not truncate."
            ),
        },
    },
}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": TASK_DEFINITIONS,
        "action_schema": ACTION_SCHEMA,
        "total_tasks": len(TASK_DEFINITIONS),
        "difficulty_range": "easy → medium → hard",
    }


# ------------------------------------------------------------------ #
#  /grader  — score a fix without running a full episode              #
# ------------------------------------------------------------------ #

class GraderRequest(BaseModel):
    task_id: str
    bug_type: str
    diagnosis: str
    fixed_code: str
    seed: int = 42


@app.post("/grader")
def run_grader(req: GraderRequest) -> Dict[str, Any]:
    valid_tasks = [TASK_SHAPE_MISMATCH, TASK_TRAINING_COLLAPSE, TASK_DATA_LEAKAGE]
    if req.task_id not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"task_id must be one of {valid_tasks}",
        )
    try:
        scenario = get_scenario(req.task_id, seed=req.seed)
        result = grade(
            action_bug_type=req.bug_type,
            action_diagnosis=req.diagnosis,
            fixed_code=req.fixed_code,
            scenario=scenario,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "task_id": req.task_id,
        "score": result.score,
        "feedback": result.feedback,
        "execution_output": result.execution_output,
    }


# ------------------------------------------------------------------ #
#  /baseline  — run the Groq-powered baseline agent on all 3 tasks   #
# ------------------------------------------------------------------ #

@app.get("/baseline")
async def run_baseline() -> Dict[str, Any]:
    """
    Runs the Groq-based baseline agent against all 3 tasks and returns scores.
    Requires GROQ_API_KEY environment variable.
    """
    groq_api_key = (os.environ.get("API_KEY") or os.environ.get("GROQ_API_KEY", "")).strip()
    if not groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="API_KEY or GROQ_API_KEY environment variable not set.",
        )

    try:
        server_dir = os.path.dirname(os.path.abspath(__file__))
        if server_dir not in sys.path:
            sys.path.insert(0, server_dir)
        from baseline_inference import run_baseline_on_all_tasks
        base_url = (os.environ.get("API_BASE_URL") or "https://api.groq.com/openai/v1").strip()
        results = await asyncio.get_event_loop().run_in_executor(
            None, run_baseline_on_all_tasks, groq_api_key, base_url
        )
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Baseline run failed: {e}\n{traceback.format_exc()}")

    avg = sum(r["score"] for r in results) / len(results) if results else 0.0

    return {
        "results": results,
        "average_score": round(avg, 4),
        "model": "llama-3.3-70b-versatile (Groq)",
        "note": "Baseline uses a single-shot zero-prompt strategy with no examples.",
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
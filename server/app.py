import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import asyncio
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openenv.core.env_server.http_server import create_app

from models import DebugAction, DebugObservation, DebugState
from ml_debug_env_environment import MlDebugEnvEnvironment
from bug_generator import (
    TASK_SHAPE_MISMATCH,
    TASK_TRAINING_COLLAPSE,
    TASK_DATA_LEAKAGE,
    TASK_WRONG_DEVICE,
    TASK_GRADIENT_NOT_ZEROED,
    TASK_MISSING_EVAL_MODE,
    TASK_COMPOUND_SHAPE_DEVICE,
    TASK_COMPOUND_LEAKAGE_EVAL,
    get_scenario,
)
from grader import grade

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

TASK_DEFINITIONS = [
    {
        "task_id": TASK_SHAPE_MISMATCH,
        "name": "Shape Mismatch",
        "difficulty": "easy",
        "num_bugs": 1,
        "description": (
            "A PyTorch model crashes immediately with a RuntimeError during the forward pass. "
            "The architectural bug is explicit in the traceback. "
            "Fix the script so it trains for 3 epochs without error."
        ),
        "success_criteria": "Code runs to completion; epoch logs print; no RuntimeError.",
    },
    {
        "task_id": TASK_TRAINING_COLLAPSE,
        "name": "Training Collapse",
        "difficulty": "medium",
        "num_bugs": 1,
        "description": (
            "A PyTorch training script runs without crashing but the model completely fails to learn. "
            "Loss diverges to NaN or plateaus immediately. "
            "Fix the training bug so loss decreases consistently across all epochs."
        ),
        "success_criteria": "Loss decreases across epochs; no NaN values in output.",
    },
    {
        "task_id": TASK_WRONG_DEVICE,
        "name": "Wrong Device",
        "difficulty": "medium",
        "num_bugs": 1,
        "description": (
            "A PyTorch script crashes on the first forward pass because the model and data tensors "
            "are on different devices. Fix tensor placement so training runs cleanly."
        ),
        "success_criteria": "All tensors on same device; training completes 3 epochs without RuntimeError.",
    },
    {
        "task_id": TASK_GRADIENT_NOT_ZEROED,
        "name": "Gradient Not Zeroed",
        "difficulty": "medium-hard",
        "num_bugs": 1,
        "description": (
            "A PyTorch training script runs but loss explodes after the first epoch and collapses to NaN. "
            "No crash occurs. There is a fundamental error in the training loop structure. "
            "Fix the loop so loss decreases consistently."
        ),
        "success_criteria": "Loss decreases consistently; no NaN values; optimizer.zero_grad() before backward.",
    },
    {
        "task_id": TASK_DATA_LEAKAGE,
        "name": "Silent Data Leakage",
        "difficulty": "hard",
        "num_bugs": 1,
        "description": (
            "A PyTorch training script runs cleanly and reports impressive metrics. "
            "But the evaluation is fundamentally invalid due to a data pipeline mistake. "
            "Find the data leakage bug and fix it so the evaluation reflects true generalisation."
        ),
        "success_criteria": "Normalization stats from training data only; metrics reflect genuine generalisation.",
    },
    {
        "task_id": TASK_MISSING_EVAL_MODE,
        "name": "Missing Eval Mode",
        "difficulty": "hard",
        "num_bugs": 1,
        "description": (
            "A PyTorch classifier trains successfully but produces unstable and unreliable metrics. "
            "Running evaluation multiple times gives different results. "
            "Fix the evaluation so it produces stable, deterministic metrics."
        ),
        "success_criteria": "model.eval() and torch.no_grad() during evaluation; identical results on repeated runs.",
    },
    {
        "task_id": TASK_COMPOUND_SHAPE_DEVICE,
        "name": "Compound: Shape + Device",
        "difficulty": "medium-hard",
        "num_bugs": 2,
        "description": (
            "This script has TWO bugs that must both be fixed: "
            "a shape mismatch in the model architecture AND a device placement error. "
            "Fix both bugs so the script trains for 3 epochs without any errors."
        ),
        "success_criteria": "Both shape mismatch and device mismatch resolved; training completes cleanly.",
    },
    {
        "task_id": TASK_COMPOUND_LEAKAGE_EVAL,
        "name": "Compound: Leakage + Eval Mode",
        "difficulty": "expert",
        "num_bugs": 2,
        "description": (
            "This script has TWO silent bugs that make the evaluation invalid: "
            "a data leakage bug in preprocessing AND a missing eval mode bug. "
            "Fix both so the evaluation is correct and deterministic."
        ),
        "success_criteria": "Train-only normalization stats; model.eval() during eval; deterministic and realistic metrics.",
    },
]

ACTION_SCHEMA = {
    "type": "object",
    "required": ["bug_type", "diagnosis", "fixed_code"],
    "properties": {
        "bug_type": {
            "type": "string",
            "description": "Category of bug(s) identified.",
            "enum": [
                "shape_mismatch",
                "training_collapse",
                "data_leakage",
                "wrong_device",
                "gradient_not_zeroed",
                "missing_eval_mode",
                "compound_shape_device",
                "compound_leakage_eval",
                "other",
            ],
        },
        "diagnosis": {
            "type": "string",
            "description": "Plain-language explanation of the root cause(s).",
        },
        "fixed_code": {
            "type": "string",
            "description": "Complete corrected Python script. Runnable as-is. All imports included.",
        },
    },
}

VALID_TASK_IDS = [t["task_id"] for t in TASK_DEFINITIONS]


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": TASK_DEFINITIONS,
        "action_schema": ACTION_SCHEMA,
        "total_tasks": len(TASK_DEFINITIONS),
        "difficulty_range": "easy → medium → medium-hard → hard → expert",
        "compound_tasks": [TASK_COMPOUND_SHAPE_DEVICE, TASK_COMPOUND_LEAKAGE_EVAL],
        "note": "Compound tasks contain TWO bugs that must both be fixed for full score.",
    }


class GraderRequest(BaseModel):
    task_id: str
    bug_type: str
    diagnosis: str
    fixed_code: str
    seed: int = 42


@app.post("/grader")
def run_grader(req: GraderRequest) -> Dict[str, Any]:
    if req.task_id not in VALID_TASK_IDS:
        raise HTTPException(status_code=400, detail=f"task_id must be one of {VALID_TASK_IDS}")
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


@app.get("/baseline")
async def run_baseline() -> Dict[str, Any]:
    api_key = (
        os.environ.get("HF_TOKEN") or
        os.environ.get("API_KEY") or
        os.environ.get("GROQ_API_KEY", "")
    ).strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="HF_TOKEN, API_KEY, or GROQ_API_KEY not set.")

    try:
        server_dir = os.path.dirname(os.path.abspath(__file__))
        if server_dir not in sys.path:
            sys.path.insert(0, server_dir)
        from baseline_inference import run_baseline_on_all_tasks
        base_url = (os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1").strip()
        results = await asyncio.get_event_loop().run_in_executor(
            None, run_baseline_on_all_tasks, api_key, base_url
        )
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Baseline run failed: {e}\n{traceback.format_exc()}")

    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {
        "results": results,
        "average_score": round(avg, 4),
        "model": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
        "note": "Baseline uses multi-turn retry with grader feedback.",
    }

@app.get("/", response_class=HTMLResponse)
def landing_page():
    html_path = os.path.join(os.path.dirname(__file__), "landing_page.html")
    with open(html_path) as f:
        return f.read()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
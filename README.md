---
title: ML Debug Env
emoji: 🐛
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
app_port: 8000
tags:
  - openenv
  - pytorch
  - reinforcement-learning
  - debugging
  - llm
---

# 🐛 ML Debug Env

> **Meta × PyTorch × Scaler OpenEnv Hackathon — April 2026**

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) reinforcement-learning environment where AI agents debug broken PyTorch training scripts. The agent receives a buggy Python script plus its failure output, then must return a **corrected script**. The grader executes the fix in an isolated subprocess and scores it **0.01 → 0.99** with partial credit at every verification stage.

---

## 🧠 What Is This?

Most ML benchmarks test whether a model can *write* code. This environment tests something harder: can a model *debug* code?

The agent is dropped into a broken PyTorch training environment. It sees the buggy script, the error or behavioral failure, and must:
1. Identify what category of bug is present
2. Explain the root cause in plain English
3. Return a complete, corrected, runnable Python script

The grader then **actually executes** the fixed code in a subprocess and scores it across 6 stages of increasing correctness. There is no cheating — the code has to run.

This is a realistic simulation of the debugging loop every ML engineer goes through daily.

---

## 🎯 Tasks

Three tasks of increasing difficulty, each representing a common class of real-world ML bugs:

| task_id | Difficulty | What's Broken | Why It's Hard |
|---|---|---|---|
| `shape_mismatch` | 🟢 Easy | `nn.Linear` input dim is wrong → explicit `RuntimeError` crash | Error is in the traceback. Read it. |
| `training_collapse` | 🟡 Medium | Huge LR → NaN loss, **or** wrong loss fn → flat plateau. No crash. | No error — agent must reason about training dynamics |
| `data_leakage` | 🔴 Hard | Dataset normalized before train/test split → inflated metrics, no error | Everything *looks* fine. Agent must understand data pipelines |

### Task 1 — Shape Mismatch (Easy)

A `SimpleClassifier` has a 2-layer encoder that outputs `hidden_size` features (e.g. 256), but the classifier head is initialized with `nn.Linear(wrong_size, num_classes)` where `wrong_size` (e.g. 32) doesn't match. The forward pass crashes immediately with:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (256 cannot be broadcast to 32)
```

The fix: change `nn.Linear(wrong_size, ...)` to `nn.Linear(hidden_size, ...)`.

The `hidden_size` and `wrong_size` values are randomized per episode seed so agents can't hardcode the fix.

### Task 2 — Training Collapse (Medium)

Two variants, randomly selected:

**Variant A — Bad learning rate:** SGD with `lr=100.0` (or 10.0 or 50.0). Loss explodes to NaN by epoch 2. Code runs, no crash, just broken training. Fix: reduce lr to ~1e-3.

**Variant B — Wrong loss function:** Binary classification model with sigmoid output trained with `MSELoss` instead of `BCELoss`. Loss plateaus immediately around 0.25 and never decreases. Fix: switch to `BCELoss` or remove sigmoid and use `BCEWithLogitsLoss`.

The agent must distinguish between these variants from the error description alone.

### Task 3 — Data Leakage (Hard)

Two variants, randomly selected:

**Variant A:** Normalization computed on full dataset (`X_raw.mean(dim=0)`) before the train/test split. Test set statistics contaminate training. Reported accuracy looks great (~96%) but is invalid.

**Variant B:** Train/test split done first, but `full_mean = X_raw.mean(dim=0)` still computed on the whole dataset and used to normalize both sets. Same bug, different code structure.

Fix: compute `mean` and `std` only from `X_train`, then apply those stats to normalize both `X_train` and `X_test`.

This task has no error, no NaN, no crash. The agent must reason about data flow.

---

## 🏆 Scoring Ladder

The grader runs the agent's fixed code in a real subprocess and scores it across 6 stages:

| Score | Stage | Condition |
|---|---|---|
| **0.01** | Bug type wrong | Agent identified the wrong bug category |
| **0.2** | Code crashes | Right bug type, but fixed code throws an exception |
| **0.4** | Incomplete training | Code runs but training loop doesn't finish |
| **0.6** | Root cause not fixed | Training completes but the underlying bug is still present |
| **0.8** | Success signal missing | Fix is valid but expected output pattern not found |
| **0.99** | Perfect fix | Code runs, training finishes, success signal confirmed ✅ |

Scores are intentionally kept off 0.0 and 1.0 to satisfy the validator's strict `(0, 1)` exclusive range requirement.

---

## 📐 Action Space

The agent returns a single JSON object:

```json
{
  "bug_type": "shape_mismatch",
  "diagnosis": "The classifier head uses nn.Linear(32, 10) but the encoder outputs 256 features. The input dimension must match the encoder output.",
  "fixed_code": "import torch\nimport torch.nn as nn\n..."
}
```

| Field | Type | Values |
|---|---|---|
| `bug_type` | `str` | `shape_mismatch`, `training_collapse`, `data_leakage`, `other` |
| `diagnosis` | `str` | Plain-language explanation of the root cause |
| `fixed_code` | `str` | Complete corrected Python script, all imports included, runnable as-is |

---

## 👁 Observation Space

What the agent receives on `reset()` and after each `step()`:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Which task is active |
| `task_description` | `str` | Natural language description of what to fix |
| `buggy_code` | `str` | The broken Python script |
| `error_output` | `str` | stderr / traceback or behavioral failure description |
| `execution_result` | `str \| None` | stdout+stderr from running the agent's fix (`None` on reset) |
| `grader_score` | `float \| None` | Score 0.01–0.99 (`None` on reset) |
| `grader_feedback` | `str \| None` | Human-readable explanation of the score |
| `step_number` | `int` | Current step within this episode |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float \| None` | Same as `grader_score` |

---

## 🚀 Quick Start

### Install the client

```bash
pip install git+https://huggingface.co/spaces/rak2315/ml-debug-env
```

### Async usage

```python
import asyncio
from ml_debug_env import MlDebugEnvClient, DebugAction

SPACE_URL = "https://rak2315-ml-debug-env.hf.space"

async def main():
    async with MlDebugEnvClient(base_url=SPACE_URL) as env:
        result = await env.reset()
        obs = result.observation

        print(f"Task: {obs.task_id}")
        print(f"Description: {obs.task_description}")
        print(f"Error:\n{obs.error_output}\n")

        action = DebugAction(
            bug_type="shape_mismatch",
            diagnosis="nn.Linear classifier input dim is 32 but encoder outputs 256",
            fixed_code=obs.buggy_code.replace("nn.Linear(32,", "nn.Linear(256,"),
        )

        result = await env.step(action)
        print(f"Score: {result.observation.grader_score}")
        print(f"Feedback: {result.observation.grader_feedback}")

asyncio.run(main())
```

### Sync usage

```python
from ml_debug_env import MlDebugEnvClient, DebugAction

SPACE_URL = "https://rak2315-ml-debug-env.hf.space"

with MlDebugEnvClient(base_url=SPACE_URL).sync() as env:
    result = env.reset()
    obs = result.observation

    result = env.step(DebugAction(
        bug_type="training_collapse",
        diagnosis="Learning rate 100.0 causes gradient explosion leading to NaN loss",
        fixed_code=obs.buggy_code.replace("lr=100.0", "lr=1e-3"),
    ))
    print(result.observation.grader_score)
```

---

## 🌐 API Endpoints

The server exposes both the standard OpenEnv spec endpoints and additional utility endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new debug episode, returns buggy code + error |
| `POST` | `/step` | Submit a fix attempt, returns score + feedback |
| `GET` | `/state` | Get current episode state |
| `GET` | `/health` | Health check — `{"status": "healthy"}` |
| `GET` | `/tasks` | List all 3 tasks with descriptions and action schema |
| `POST` | `/grader` | Score a fix directly without running a full episode |
| `GET` | `/baseline` | Run the built-in LLM baseline agent on all 3 tasks |
| `WS` | `/ws` | Persistent WebSocket session for low-latency interaction |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/schema` | Environment JSON schema |

---

## 📊 Baseline Results

Built-in baseline agent: `llama-3.3-70b-versatile` — zero-shot, single attempt, no examples.

| task_id | Score |
|---|---|
| `shape_mismatch` | **0.99** |
| `training_collapse` | **0.99** |
| `data_leakage` | **0.99** |
| **Average** | **0.99** |

The baseline uses a structured JSON system prompt and the OpenAI-compatible Groq API. It correctly identifies the bug type and returns a complete fixed script for all 3 tasks in a single zero-shot pass.

---

## 🏗 Architecture

```
Agent / Validator
       │
       ▼
  FastAPI Server (server/app.py)
       │
       ├── POST /reset ──► MlDebugEnvEnvironment.reset()
       │                         │
       │                         ▼
       │                   BugGenerator → BugScenario
       │                   (buggy code + error output)
       │
       ├── POST /step ──► MlDebugEnvEnvironment.step()
       │                         │
       │                         ▼
       │                   Grader.grade()
       │                         │
       │                   subprocess.run(fixed_code)
       │                         │
       │                   Score 0.01 → 0.99
       │
       └── GET /baseline ─► BaselineInference
                                 │
                             OpenAI Client
                                 │
                             LLM Proxy (API_BASE_URL)
                                 │
                             llama-3.3-70b-versatile
```

---

## 📁 Project Structure

```
ml_debug_env/
├── inference.py                     # Validator entry point — calls LLM proxy directly
├── __init__.py                      # Package exports
├── models.py                        # Pydantic models: DebugAction, DebugObservation, DebugState
├── client.py                        # MlDebugEnvClient (EnvClient subclass)
├── Dockerfile                       # Root Dockerfile for HF Space deployment
├── openenv.yaml                     # OpenEnv manifest
├── pyproject.toml                   # Project metadata + deps
├── baseline_results.json            # Latest baseline run results
├── test.py                          # Sanity tests (no server needed)
├── test2.py                         # Full 20-check submission validator
└── server/
    ├── app.py                       # FastAPI app — all endpoints
    ├── bug_generator.py             # Procedural buggy script generation
    ├── grader.py                    # Executes fixed code, returns 0.01–0.99 score
    ├── ml_debug_env_environment.py  # OpenEnv Environment class (reset/step/state)
    ├── baseline_inference.py        # LLM baseline agent
    ├── Dockerfile                   # Alternative Dockerfile (used for local dev)
    └── requirements.txt             # Server runtime deps
```

### File Responsibilities

**`server/bug_generator.py`**
Procedurally generates buggy PyTorch scripts using a random seed. Each task has randomized parameters (hidden sizes, learning rates, variants) so the same task_id produces slightly different code each episode. Provides `get_scenario(task_id, seed)` → `BugScenario`.

**`server/grader.py`**
The referee. Writes the agent's fixed code to a temp file, executes it via `subprocess.run` using the correct Python interpreter (found via `PYTHON_EXEC` env var or venv detection), and applies 5 verification stages. Each stage checks progressively deeper correctness — from "does it run" to "does the output contain the right success signal".

**`server/ml_debug_env_environment.py`**
Implements the OpenEnv `Environment` interface. Manages episode state, rotates through tasks when none is pinned, handles `reset()` / `step()` / `state` with proper session isolation. Supports concurrent sessions.

**`server/app.py`**
FastAPI application. Wraps the environment with the OpenEnv `create_app()` factory, then adds `/tasks`, `/grader`, and `/baseline` endpoints on top. The `/baseline` endpoint runs the LLM agent on all 3 tasks and returns scores.

**`server/baseline_inference.py`**
The built-in LLM agent. Uses the OpenAI client pointed at `API_BASE_URL` (Groq by default). Sends buggy code + error to `llama-3.3-70b-versatile` with a structured JSON system prompt. Parses the response and passes the fix to the grader.

**`inference.py`** (root)
What the hackathon validator actually executes. Reads `API_BASE_URL`, `MODEL_NAME`, `API_KEY` from environment, calls the LLM proxy directly (not through the server), grades results locally, and emits structured stdout logs in the required format:
```
[START] task=shape_mismatch
[STEP] step=1 reward=0.9900
[END] task=shape_mismatch score=0.9900 steps=1
```

**`models.py`**
Pydantic v2 models for the action/observation/state types used across the whole system.

**`client.py`**
Python client library so external agents can interact with the deployed HF Space using the `MlDebugEnvClient` class instead of raw HTTP.

---

## 🛠 Local Development

```bash
git clone https://huggingface.co/spaces/rak2315/ml-debug-env
cd ml-debug-env

# Install deps
pip install -e .

# Set env vars
set GROQ_API_KEY=your_groq_key
set PYTHON_EXEC=path/to/python/with/torch  # Windows: needed if venv lacks torch

# Run server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Run full validation (20 checks)
python test2.py
```

**With Docker:**

```bash
docker build -t ml-debug-env:latest .
docker run -e GROQ_API_KEY=your_key -p 8000:8000 ml-debug-env:latest
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Injected by validator | LLM proxy endpoint. Falls back to Groq if not set. |
| `API_KEY` | Injected by validator | LLM proxy API key. Falls back to `GROQ_API_KEY`. |
| `MODEL_NAME` | Injected by validator | Model to use. Defaults to `llama-3.3-70b-versatile`. |
| `GROQ_API_KEY` | Optional (local dev) | Enables `/baseline` endpoint locally via Groq free tier. |
| `PYTHON_EXEC` | Optional (local dev) | Path to Python interpreter with torch installed, used by grader subprocess. |
| `HF_TOKEN` | Optional | HuggingFace token for Space access. |

---

## 📄 License

BSD 3-Clause — see [OpenEnv](https://github.com/meta-pytorch/OpenEnv/blob/main/LICENSE)
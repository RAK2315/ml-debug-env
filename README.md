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

The environment also supports **multi-turn episodes** — if the agent's first fix is incomplete, the grader's feedback is fed back and the agent gets up to 3 attempts to reach a perfect score. This simulates the real debugging loop every ML engineer goes through daily.

---

## 🎯 Tasks

Six tasks of increasing difficulty, covering the most common classes of real-world PyTorch bugs:

| task_id | Difficulty | What's Broken | Why It's Hard |
|---|---|---|---|
| `shape_mismatch` | 🟢 Easy | `nn.Linear` input dim is wrong → explicit `RuntimeError` crash | Error is in the traceback. Read it. |
| `training_collapse` | 🟡 Medium | Huge LR → NaN loss, **or** wrong loss fn → flat plateau. No crash. | No error — agent must reason about training dynamics |
| `wrong_device` | 🟡 Medium | Model on CUDA, data on CPU → explicit `RuntimeError` on forward pass | Device mismatch is in the traceback but fix requires understanding tensor placement |
| `gradient_not_zeroed` | 🟠 Medium-Hard | `optimizer.zero_grad()` missing → gradients accumulate → loss explodes to NaN | No crash, no error. Agent must reason about training loop structure |
| `data_leakage` | 🔴 Hard | Dataset normalized before train/test split → inflated metrics, no error | Everything *looks* fine. Agent must understand data pipelines |
| `missing_eval_mode` | 🔴 Hard | `model.eval()` and `torch.no_grad()` missing → Dropout active during eval → unstable metrics | No error, code runs. Agent must understand train vs eval mode semantics |

---

### Task 1 — Shape Mismatch (Easy)

A `SimpleClassifier` has a 2-layer encoder that outputs `hidden_size` features (e.g. 256), but the classifier head is initialized with `nn.Linear(wrong_size, num_classes)` where `wrong_size` doesn't match. The forward pass crashes immediately with:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (256 cannot be broadcast to 32)
```

The `hidden_size` and `wrong_size` values are randomized per episode seed.

---

### Task 2 — Training Collapse (Medium)

Two variants, randomly selected:

**Variant A — Bad learning rate:** SGD with `lr=100.0` (or 10.0 or 50.0). Loss explodes to NaN by epoch 2. Fix: reduce lr to ~1e-3.

**Variant B — Wrong loss function:** Binary classification model with sigmoid output trained with `MSELoss` instead of `BCELoss`. Loss plateaus immediately. Fix: switch to `BCELoss` or use `BCEWithLogitsLoss`.

---

### Task 3 — Wrong Device (Medium)

Model is moved to GPU via `.to(device)` but data batches remain on CPU. Every forward pass crashes with:

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

Fix: move `xb` and `yb` to device inside the training loop.

---

### Task 4 — Gradient Not Zeroed (Medium-Hard)

`optimizer.zero_grad()` is missing from the training loop. Gradients from previous batches accumulate, causing loss to explode after the first epoch and collapse to NaN. No crash occurs. The agent must identify the missing call from loss behavior alone.

---

### Task 5 — Data Leakage (Hard)

Two variants, randomly selected:

**Variant A:** Normalization computed on full dataset before the train/test split. Test set statistics contaminate training.

**Variant B:** Split done first, but `full_mean = X_raw.mean(dim=0)` still computed on whole dataset and used to normalize both sets.

Fix: compute `mean` and `std` only from `X_train`, then apply those stats to both sets.

---

### Task 6 — Missing Eval Mode (Hard)

A classifier with `Dropout` and `BatchNorm` layers is evaluated without calling `model.eval()` or `torch.no_grad()`. Dropout stays active during inference, causing different predictions each forward pass. Test accuracy is noisy and unreliable. No error is raised.

Fix: call `model.eval()` and wrap evaluation in `torch.no_grad()`.

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
| `bug_type` | `str` | `shape_mismatch`, `training_collapse`, `data_leakage`, `wrong_device`, `gradient_not_zeroed`, `missing_eval_mode`, `other` |
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

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new debug episode, returns buggy code + error |
| `POST` | `/step` | Submit a fix attempt, returns score + feedback |
| `GET` | `/state` | Get current episode state |
| `GET` | `/health` | Health check — `{"status": "healthy"}` |
| `GET` | `/tasks` | List all 6 tasks with descriptions and action schema |
| `POST` | `/grader` | Score a fix directly without running a full episode |
| `GET` | `/baseline` | Run the built-in LLM baseline agent on all 6 tasks |
| `WS` | `/ws` | Persistent WebSocket session for low-latency interaction |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/schema` | Environment JSON schema |

---

## 📊 Baseline Results

Built-in baseline agent: `llama-3.3-70b-versatile` — zero-shot, up to 3 self-correcting attempts using grader feedback.

| task_id | Difficulty | Score |
|---|---|---|
| `shape_mismatch` | Easy | **0.99** |
| `training_collapse` | Medium | **0.99** |
| `wrong_device` | Medium | **0.99** |
| `gradient_not_zeroed` | Medium-Hard | **0.99** |
| `data_leakage` | Hard | **0.99** |
| `missing_eval_mode` | Hard | **0.99** |
| **Average** | — | **0.99** |

The baseline uses a structured JSON system prompt with explicit bug type constraints and a multi-turn retry loop — if the first fix scores below 0.95, grader feedback is injected into the next attempt.

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
                             Multi-turn retry loop
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
└── server/
    ├── app.py                       # FastAPI app — all endpoints
    ├── bug_generator.py             # Procedural buggy script generation (6 tasks)
    ├── grader.py                    # Executes fixed code, returns 0.01–0.99 score
    ├── ml_debug_env_environment.py  # OpenEnv Environment class (reset/step/state)
    ├── baseline_inference.py        # LLM baseline agent with multi-turn retry
    ├── Dockerfile                   # Alternative Dockerfile (local dev)
    └── requirements.txt             # Server runtime deps
```

---

## 🛠 Local Development

```bash
git clone https://huggingface.co/spaces/rak2315/ml-debug-env
cd ml-debug-env

pip install -e .

# Set env vars (Windows)
set GROQ_API_KEY=your_groq_key
set PYTHON_EXEC=path\to\python\with\torch

# Run server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Verify all 6 tasks
curl http://localhost:8000/tasks
curl http://localhost:8000/health
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
| `API_BASE_URL` | Injected by validator | LLM proxy endpoint. Defaults to HF inference router. |
| `API_KEY` | Injected by validator | LLM proxy API key. |
| `HF_TOKEN` | Injected by validator | Primary auth token — checked before API_KEY. |
| `MODEL_NAME` | Injected by validator | Model to use. Defaults to `Qwen/Qwen2.5-72B-Instruct`. |
| `GROQ_API_KEY` | Optional (local dev) | Enables `/baseline` endpoint locally via Groq free tier. |
| `PYTHON_EXEC` | Optional (local dev) | Path to Python interpreter with torch installed, used by grader subprocess. |

---

## 📄 License

BSD 3-Clause — see [OpenEnv](https://github.com/meta-pytorch/OpenEnv/blob/main/LICENSE)
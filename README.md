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

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) reinforcement-learning environment where AI agents debug broken PyTorch training scripts. The agent receives a buggy Python script plus its failure output, then must return a **corrected script**. The grader executes the fix in an isolated subprocess and scores it **0.0 → 1.0** with partial credit at every verification stage.

---

## 🎯 Tasks

| task\_id | Difficulty | What's Broken | Success Criteria |
|---|---|---|---|
| `shape_mismatch` | 🟢 Easy | `nn.Linear` input dim is wrong → explicit `RuntimeError` crash | Trains 3 epochs, no `RuntimeError` |
| `training_collapse` | 🟡 Medium | Huge LR → NaN loss **or** wrong loss fn → flat plateau (no crash) | Loss decreases across all epochs, no NaN |
| `data_leakage` | 🔴 Hard | Dataset normalized before train/test split → inflated metrics, no error | Norm stats computed from train set only; metrics reflect true generalisation |

---

## 🏆 Scoring Ladder

| Score | Condition |
|---|---|
| **0.0** | Wrong `bug_type` identified |
| **0.2** | Correct `bug_type`, fixed code **crashes** |
| **0.4** | Correct `bug_type`, runs but training doesn't finish |
| **0.6** | Runs + finishes, root cause **not** actually fixed |
| **0.8** | Root cause fixed, success signal not confirmed in output |
| **1.0** | Perfect fix — runs, finishes, success signal confirmed ✅ |

---

## 📐 Action Space

The agent returns a single JSON action:

| Field | Type | Description |
|---|---|---|
| `bug_type` | `str` | One of: `shape_mismatch`, `training_collapse`, `data_leakage`, `other` |
| `diagnosis` | `str` | Plain-language explanation of the root cause |
| `fixed_code` | `str` | Complete corrected Python script, runnable as-is (all imports included) |

## 👁 Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Active task identifier |
| `task_description` | `str` | Natural language description of what to fix |
| `buggy_code` | `str` | The broken Python script |
| `error_output` | `str` | stderr / traceback or behavioural failure description |
| `execution_result` | `str \| None` | stdout+stderr from running the agent's fix (`None` on reset) |
| `grader_score` | `float \| None` | Score 0.0–1.0 (`None` on reset) |
| `grader_feedback` | `str \| None` | Human-readable explanation of the score |
| `step_number` | `int` | Current step within this episode |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float \| None` | Same as `grader_score` |

---

## 🚀 Quick Start

```bash
pip install git+https://huggingface.co/spaces/rehaan/ml-debug-env
```

```python
import asyncio
from ml_debug_env import MlDebugEnvClient, DebugAction

SPACE_URL = "https://rehaan-ml-debug-env.hf.space"

async def main():
    async with MlDebugEnvClient(base_url=SPACE_URL) as env:
        # Reset — get the buggy script
        result = await env.reset()
        obs = result.observation
        print(f"Task: {obs.task_id}")
        print(f"Error:\n{obs.error_output}\n")

        # Step — submit a fix
        action = DebugAction(
            bug_type="shape_mismatch",
            diagnosis="nn.Linear classifier input dim is 32 but encoder outputs 256",
            fixed_code=obs.buggy_code.replace(
                "nn.Linear(32,", "nn.Linear(256,"
            ),
        )
        result = await env.step(action)
        print(f"Score: {result.observation.grader_score}")
        print(f"Feedback: {result.observation.grader_feedback}")

asyncio.run(main())
```

**Synchronous usage:**

```python
from ml_debug_env import MlDebugEnvClient, DebugAction

with MlDebugEnvClient(base_url=SPACE_URL).sync() as env:
    result = env.reset()
    obs = result.observation

    result = env.step(DebugAction(
        bug_type="training_collapse",
        diagnosis="Learning rate 100.0 causes gradient explosion → NaN",
        fixed_code=obs.buggy_code.replace("lr=100.0", "lr=1e-3"),
    ))
    print(result.observation.grader_score)
```

---

## 🌐 API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new debug episode |
| `POST` | `/step` | Submit a fix attempt |
| `GET` | `/state` | Get current episode state |
| `GET` | `/health` | Health check (`{"status":"healthy"}`) |
| `GET` | `/tasks` | List all 3 tasks + action schema |
| `POST` | `/grader` | Score a fix without running a full episode |
| `GET` | `/baseline` | Run Groq llama-3.3-70b-versatile baseline on all 3 tasks |
| `WS` | `/ws` | Persistent WebSocket session (low-latency) |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/schema` | Environment JSON schema |

---

## 📊 Baseline Results

Baseline agent: `llama-3.3-70b-versatile` via Groq (zero-shot, single attempt)

| task\_id | Score |
|---|---|
| `shape_mismatch` | TBD |
| `training_collapse` | TBD |
| `data_leakage` | TBD |
| **Average** | **TBD** |

> Run `/baseline` endpoint with `GROQ_API_KEY` set to populate these.

---

## 🛠 Local Development

```bash
git clone https://huggingface.co/spaces/rehaan/ml-debug-env
cd ml-debug-env

# Install deps
pip install -e .

# Run server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Quick sanity test
python test.py
```

**With Docker:**

```bash
docker build -f server/Dockerfile -t ml-debug-env:latest .
docker run -e GROQ_API_KEY=your_key -p 8000:8000 ml-debug-env:latest
```

---

## 📁 Project Structure

```
ml_debug_env/
├── __init__.py                      # Package exports
├── models.py                        # DebugAction, DebugObservation, DebugState
├── client.py                        # MlDebugEnvClient (EnvClient subclass)
├── openenv.yaml                     # OpenEnv manifest
├── pyproject.toml                   # Project metadata + deps
├── test.py                          # Sanity tests (no server needed)
└── server/
    ├── app.py                       # FastAPI app (create_app + extra endpoints)
    ├── bug_generator.py             # Generates buggy scripts with known fixes
    ├── grader.py                    # Executes fixed code, returns 0.0–1.0 score
    ├── ml_debug_env_environment.py  # MlDebugEnvEnvironment (reset/step/state)
    ├── baseline_inference.py        # Groq llama-3.3-70b-versatile baseline
    ├── Dockerfile                   # python:3.10-slim + CPU torch
    └── requirements.txt             # Server runtime deps
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Optional | Enables `/baseline` endpoint (Groq free tier) |

---

## 📄 License

BSD 3-Clause — see [OpenEnv](https://github.com/meta-pytorch/OpenEnv/blob/main/LICENSE)

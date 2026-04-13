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

Most ML benchmarks test whether a model can *write* code from scratch. This environment tests something harder and more realistic: can a model *debug* code that someone else wrote?

Every ML engineer spends a significant chunk of their day doing exactly this — staring at a broken training script, reading error messages, checking loss curves, and figuring out why the model isn't learning. It's a skill that requires understanding not just Python syntax but the deeper mechanics of how neural networks train.

This environment puts an AI agent in that exact situation. The agent sees the buggy script and the failure output, and must:

1. **Identify** what category of bug is present (is it a crash? a silent training failure? a data pipeline problem?)
2. **Explain** the root cause in plain English (why is this happening?)
3. **Return** a complete, corrected, runnable Python script (not a patch — the whole thing)

The grader then **actually executes** the fixed code in a real subprocess. There is no regex matching, no string comparison, no shortcuts — the code has to run, the training has to complete, and the output has to look right. A fix that runs but doesn't actually solve the underlying bug gets partial credit, not full marks.

The environment also supports **multi-turn episodes** — if the agent's first fix is incomplete, the grader's feedback is returned as the next observation and the agent gets up to 3 attempts. This simulates the real debugging loop: try something, see what happens, adjust, try again.

---

## 🎯 Tasks

Six tasks of increasing difficulty, each representing a distinct class of real-world PyTorch bug:

| task_id | Difficulty | What's Broken | Why It's Hard |
|---|---|---|---|
| `shape_mismatch` | 🟢 Easy | `nn.Linear` input dim is wrong → explicit `RuntimeError` crash | Error is in the traceback. Read it. |
| `training_collapse` | 🟡 Medium | Huge LR → NaN loss, **or** wrong loss fn → flat plateau. No crash. | No error — agent must reason about training dynamics |
| `wrong_device` | 🟡 Medium | Model on CUDA, data on CPU → explicit `RuntimeError` on forward pass | Device mismatch is in the traceback but fix requires understanding tensor placement |
| `gradient_not_zeroed` | 🟠 Medium-Hard | `optimizer.zero_grad()` missing → gradients accumulate → loss explodes | No crash, no error. Agent must reason about training loop structure |
| `data_leakage` | 🔴 Hard | Dataset normalized before train/test split → inflated metrics, no error | Everything *looks* fine. Agent must understand data pipelines |
| `missing_eval_mode` | 🔴 Hard | `model.eval()` and `torch.no_grad()` missing → Dropout active during eval | No error, code runs, metrics are just wrong. Agent must understand train vs eval semantics |

---

### Task 1 — Shape Mismatch (Easy)

**In plain terms:** Imagine a pipeline where the first machine outputs boxes of size 256, but the second machine is built to only accept boxes of size 32. The moment you try to pass anything through, it physically can't fit — immediate crash.

In PyTorch: a `SimpleClassifier` has an encoder that outputs `hidden_size` features, but the classifier head is initialized with the wrong input size. The forward pass crashes immediately with:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (256 cannot be broadcast to 32)
```

The `hidden_size` and `wrong_size` values are **randomized per episode seed** — agents can't memorize the answer, they must read the error and fix the correct number.

**Correct fix:** Change the classifier's input dimension to match what the encoder actually outputs.

---

### Task 2 — Training Collapse (Medium)

**In plain terms:** The script runs without crashing, but the model never learns anything. Either the loss explodes to infinity (NaN) or sits completely flat. No error message — you watch the numbers and realize something is deeply wrong.

Two variants, randomly selected per episode:

**Variant A — Bad learning rate:** SGD with `lr=100.0`. Think of learning rate as step size when walking downhill. A step of 100 means you leap so far you fly off the mountain entirely. Loss explodes to NaN by epoch 2. Fix: reduce lr to ~1e-3.

**Variant B — Wrong loss function:** Binary classification (yes/no output) trained with `MSELoss` (designed for continuous regression) instead of `BCELoss` (designed for yes/no). Loss plateaus at ~0.25 and never moves. Fix: switch to `BCELoss`.

---

### Task 3 — Wrong Device (Medium)

**In plain terms:** The model is loaded onto the GPU to run fast, but the data is still on the CPU. GPU and CPU can't directly do math together — they're in different memory spaces. PyTorch refuses and crashes immediately.

```
RuntimeError: Expected all tensors to be on the same device, but found cuda:0 and cpu!
```

**Correct fix:** Move `xb` and `yb` to the same device as the model inside the training loop using `.to(device)`.

---

### Task 4 — Gradient Not Zeroed (Medium-Hard)

**In plain terms:** In PyTorch, the gradient signals used to update model weights accumulate by default — they stack up from batch to batch unless you explicitly reset them with `optimizer.zero_grad()`. Forget this call and by epoch 2 the accumulated gradients are so large the weight updates become massive and everything collapses to NaN.

No crash, no error message. The only signal is loss behaving erratically — spiking in epoch 1, then going NaN. The agent has to recognize this pattern and trace it back to the missing one-liner.

**Correct fix:** Add `optimizer.zero_grad()` as the first line inside the batch loop, before `loss.backward()`.

---

### Task 5 — Data Leakage (Hard)

**In plain terms:** You build a model and test it on "unseen" data. But if you accidentally used statistics from the test data (its mean and standard deviation) when preprocessing everything, the model has effectively peeked at the test set. Reported accuracy looks great (~96%) but it's a lie. On truly unseen data the model would perform much worse.

Two variants, both subtle:

**Variant A:** Normalization computed on the full dataset *before* the train/test split.

**Variant B:** Split done first, but `full_mean = X_raw.mean(dim=0)` still computed on the entire dataset and used to normalize both sets.

No error, no NaN, everything runs fine. The numbers just can't be trusted.

**Correct fix:** Compute `mean` and `std` only from `X_train`, then use those same stats to normalize both sets.

---

### Task 6 — Missing Eval Mode (Hard)

**In plain terms:** PyTorch models have two modes — training and evaluation. In training mode, `Dropout` randomly zeros out neurons (prevents overfitting). During evaluation you want deterministic behavior — same input, same output, every time. Forget to switch modes and Dropout keeps randomly dropping neurons, so every evaluation run gives slightly different numbers. Your accuracy metric is meaningless noise.

The script runs fine. It prints an accuracy number. Run it again — different number. No error is raised.

**Correct fix:** Call `model.eval()` before evaluation and wrap inference in `torch.no_grad()`.

---

## 🏆 Scoring Ladder

The grader runs the agent's fixed code in a real subprocess and scores it across 6 progressive stages:

| Score | Condition | What This Means in Plain Terms |
|---|---|---|
| **0.01** | Wrong bug type | Agent looked at the code and guessed the wrong category of problem |
| **0.2** | Code crashes | Agent got the category right but the "fix" itself throws a Python error |
| **0.4** | Training doesn't complete | Code runs without crashing but training never finishes (NaN, etc.) |
| **0.6** | Root cause not fixed | Training completes but the actual underlying bug is still present |
| **0.8** | Success signal missing | Fix is genuinely valid but expected output pattern isn't in stdout |
| **0.99** | Perfect fix | Code runs, training finishes, grader confirms bug is resolved ✅ |

Scores are intentionally kept off 0.0 and 1.0 to satisfy the OpenEnv validator's strict `(0, 1)` exclusive range requirement.

**Why partial credit matters:** An agent that correctly identifies the bug but writes slightly broken Python still demonstrates useful understanding. Dense reward signals like this are more informative for RL training than binary pass/fail — the agent gets meaningful gradient signal at every stage of correctness.

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

| Field | Type | Valid Values |
|---|---|---|
| `bug_type` | `str` | `shape_mismatch`, `training_collapse`, `data_leakage`, `wrong_device`, `gradient_not_zeroed`, `missing_eval_mode`, `other` |
| `diagnosis` | `str` | Plain-language explanation of the root cause |
| `fixed_code` | `str` | The **complete** corrected Python script. All imports included. Runnable as-is. |

`bug_type` must be one of the exact strings above. The grader uses this for stage 1 scoring before running any code.

---

## 👁 Observation Space

What the agent receives from `reset()` and after each `step()`:

| Field | Type | When Set | Description |
|---|---|---|---|
| `task_id` | `str` | Always | Which of the 6 tasks is active |
| `task_description` | `str` | Always | Natural language description of the bug and what a correct fix looks like |
| `buggy_code` | `str` | Always | The full broken Python script |
| `error_output` | `str` | Always | Traceback, or behavioral failure description if no crash |
| `execution_result` | `str \| None` | After step | Full stdout+stderr from running the agent's fixed code |
| `grader_score` | `float \| None` | After step | Score 0.01–0.99 for this attempt |
| `grader_feedback` | `str \| None` | After step | Exact explanation of why this score was assigned |
| `step_number` | `int` | Always | Which attempt this is (1, 2, or 3) |
| `done` | `bool` | Always | True if score ≥ 0.95 or max steps reached |
| `reward` | `float \| None` | After step | Same value as `grader_score` |

`grader_feedback` is key for multi-turn episodes — the agent reads exactly what went wrong and uses it to write a better fix on the next attempt.

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
| `POST` | `/reset` | Start a new debug episode — returns buggy code + error output |
| `POST` | `/step` | Submit a fix — returns grader score + feedback + execution output |
| `GET` | `/state` | Current episode state (task, step count, best score) |
| `GET` | `/health` | Health check — `{"status": "healthy"}` |
| `GET` | `/tasks` | All 6 tasks with descriptions, difficulty, success criteria, action schema |
| `POST` | `/grader` | Score a fix directly without running a full episode |
| `GET` | `/baseline` | Run built-in LLM agent on all 6 tasks, return scores |
| `WS` | `/ws` | Persistent WebSocket session for low-latency interaction |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/schema` | Full environment JSON schema |

---

## 📊 Baseline Results

Built-in baseline: `llama-3.3-70b-versatile` via Groq — structured JSON prompt, up to 3 self-correcting attempts using grader feedback.

| task_id | Difficulty | Score |
|---|---|---|
| `shape_mismatch` | Easy | **0.99** |
| `training_collapse` | Medium | **0.99** |
| `wrong_device` | Medium | **0.99** |
| `gradient_not_zeroed` | Medium-Hard | **0.99** |
| `data_leakage` | Hard | **0.99** |
| `missing_eval_mode` | Hard | **0.99** |
| **Average** | — | **0.99** |

---

## 🏗 How It All Works

```
Agent
  │
  │  POST /reset
  ▼
FastAPI Server (server/app.py)
  │
  ├── MlDebugEnvEnvironment.reset()
  │       │
  │       ▼
  │   BugGenerator.get_scenario(task_id, seed)
  │       └── Returns buggy_code + error_output
  │           (solution_hint is hidden from agent)
  │
  │  POST /step  {bug_type, diagnosis, fixed_code}
  │
  ├── MlDebugEnvEnvironment.step(action)
  │       │
  │       ▼
  │   Grader.grade(action, scenario)
  │       │
  │   Stage 1: Is bug_type correct?
  │   Stage 2: Write fixed_code → temp .py → subprocess.run()
  │   Stage 3: Did it run without crashing?
  │   Stage 4: Did training complete? (no NaN, epoch logs present)
  │   Stage 5: Does fix address root cause? (code heuristics)
  │   Stage 6: Does stdout contain expected success signal?
  │       │
  │       └── Returns score (0.01–0.99) + feedback + exec_output
  │
  └── GET /baseline
          │
      baseline_inference.py
          │
      OpenAI client → API_BASE_URL → LLM
          │
      If score < 0.95: inject grader feedback → retry (max 3x)
```

---

## 📁 What Every File Does

### `inference.py` — Validator Entry Point
The file the hackathon organizers actually execute. Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`/`API_KEY` from environment (injected by the validator at eval time), calls the LLM directly, grades responses locally, and emits structured stdout logs. Runs all 6 tasks with up to 3 retry attempts each. This file never needs to be touched — it reads whatever credentials the validator injects.

### `server/bug_generator.py` — Bug Factory
Procedurally generates broken PyTorch scripts using a random seed. Each task function randomizes specific values (hidden sizes, learning rates, dropout rates, class counts) so the same bug pattern looks different each episode. `get_scenario(task_id, seed)` is the main entry point — returns a `BugScenario` with the buggy code, error description, correct bug type, and a solution hint used internally by the grader.

### `server/grader.py` — The Referee
The most critical file. Takes the agent's proposed fix and actually runs it in a subprocess. Applies 6 verification stages from "does it run" to "does the output confirm the bug is fixed." Each stage that fails returns a specific score and a feedback string. Task-specific heuristics check the fixed code itself (e.g. for data leakage: does the fixed code compute mean only from training data?). The grader is deterministic — same inputs always produce the same score.

### `server/ml_debug_env_environment.py` — Game Engine
Implements the OpenEnv `Environment` interface. Manages episode lifecycle: `reset()` picks the next task and generates a scenario, `step()` grades the action and tracks state, `state` property exposes episode metadata. Rotates through all 6 tasks when no specific task is pinned. Supports concurrent sessions.

### `server/app.py` — HTTP Server
FastAPI application. Uses OpenEnv's `create_app()` factory for standard endpoints, adds `/tasks`, `/grader`, and `/baseline` on top. The `/tasks` endpoint is particularly useful — it gives any agent a complete description of all 6 tasks and the action schema before it starts.

### `server/baseline_inference.py` — Built-in Agent
A complete LLM debugging agent used to verify the environment works and generate baseline scores. Zero-shot, structured JSON prompt, multi-turn retry using grader feedback. Intentionally simple — a floor to demonstrate correct environment behavior, not a ceiling.

### `models.py` — Type Definitions
Pydantic v2 models: `DebugAction` (what agents send), `DebugObservation` (what agents receive), `DebugState` (internal episode metadata). These are the formal interface specification of the environment.

### `client.py` — Python Client Library
Async and sync Python client for external agents to interact with the deployed HF Space without writing raw HTTP. Subclasses OpenEnv's `EnvClient`.

### `Dockerfile` (root) — HF Space Container
Packages the server for HuggingFace Spaces deployment. HF requires the Dockerfile at repo root for Docker-based Spaces.

### `openenv.yaml` — OpenEnv Manifest
Required metadata file declaring environment name, version, author, and supported task IDs. Read by `openenv validate`.

---

## 🛠 Local Development

```bash
git clone https://huggingface.co/spaces/rak2315/ml-debug-env
cd ml-debug-env

pip install -e .

# Windows
set GROQ_API_KEY=your_groq_key
set PYTHON_EXEC=C:\path\to\python\with\torch\python.exe

uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

curl http://localhost:8000/health
curl http://localhost:8000/tasks
```

**Docker:**

```bash
docker build -t ml-debug-env:latest .
docker run -e GROQ_API_KEY=your_key -p 8000:8000 ml-debug-env:latest
```

**Note on `PYTHON_EXEC`:** The grader runs fixed code in a subprocess that needs `torch` installed. Set `PYTHON_EXEC` to a Python interpreter that has torch. In Docker/HF Space this is handled automatically.

---

## 🔑 Environment Variables

| Variable | Set By | Description |
|---|---|---|
| `API_BASE_URL` | Injected by validator | LLM API endpoint. Defaults to HF inference router. |
| `HF_TOKEN` | Injected by validator | Primary auth token — checked first. |
| `API_KEY` | Injected by validator | Fallback auth token if `HF_TOKEN` not set. |
| `MODEL_NAME` | Injected by validator | Model to use. Defaults to `Qwen/Qwen2.5-72B-Instruct`. |
| `GROQ_API_KEY` | Developer | Enables `/baseline` locally via Groq free tier. |
| `PYTHON_EXEC` | Developer | Path to Python with torch. Used by grader subprocess. |

---

## 📄 License

BSD 3-Clause — see [OpenEnv](https://github.com/meta-pytorch/OpenEnv/blob/main/LICENSE)
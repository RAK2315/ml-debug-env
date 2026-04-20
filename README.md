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

> **Meta × PyTorch × Scaler OpenEnv Hackathon — April 2026** | [HF Space](https://rak2315-ml-debug-env.hf.space) | [GitHub](https://github.com/RAK2315/ml-debug-env) | [Training Notebook](https://github.com/RAK2315/ml-debug-env/blob/main/ml_debug_env_grpo.ipynb) | [Blog](https://huggingface.co/rak2315/ml-debug-env-blog)

A **partially observable** OpenEnv RL environment where AI agents debug broken PyTorch training scripts — using diagnostic tools, not oracles.

---

## 🚀 The Story: From Blind to Debugging

### Act 1: The Cold Start

The untrained agent receives its first alert:

```
"Training job crashed immediately. No epochs completed. Exit code 1."
```

It has no code. No traceback. Just a failure message and 5 steps.

It calls `view_source` immediately — brute force. Gets the full script. Pattern-matches to a fix. Scores 0.20. It got the bug type wrong.

**Reward: 0.024 average. Run 1, step 1.**

### Act 2: The Environment Fights Back

During training, something unexpected happened. The agent kept failing `shape_mismatch` fixes that looked correct. We investigated — our grader's shape verifier was too strict. It was rejecting valid fixes where the agent added an extra encoder layer to bridge the dimension gap. **The agent was right. Our environment was wrong.**

We fixed the grader. The environment improved itself because of what the agent revealed.

### Act 3: Learning a Strategy

By step 150, something changed. The agent stopped calling `view_source` first. Instead:

1. `run_code` — see what crashes
2. `inspect_gradients` — check if gradients are exploding
3. Fix — without ever viewing the source

**The inspection strategy emerged from reward signal alone. We didn't tell it to do this.**

### The Curve

![Reward Curve](https://raw.githubusercontent.com/RAK2315/ml-debug-env/main/images/reward_curve_t4_200steps.png.png)

**Run 1 (T4, 200 steps):** 0.024 → 0.190 (+0.166 improvement)

**Run 2 (H100, 500 steps — venue):** Full curve with adversarial curriculum. Updated post-hackathon.

---

## 🧠 What Makes This Different

Most debugging benchmarks hand the agent the full broken script and ask for a fix. That's not how debugging works.

**This environment is partially observable.** On `reset()`, the agent gets only:

```json
{
  "alert": "Training job failed. Final loss: nan.",
  "available_tools": ["run_code", "get_traceback", "inspect_gradients", "print_shapes", "view_source"],
  "step_budget": 5
}
```

No code. No traceback. No hints. The agent must decide what to investigate, in what order, within a 5-step budget. Every tool call costs a step. Fix correctly in 2 steps and get a 1.2× efficiency bonus. Use all 5 steps and get no bonus.

This is the loop every ML engineer goes through — run it, read the output, form a hypothesis, fix.

---

## 🎯 Tasks — 8 Tasks, Easy to Expert

| task_id | Difficulty | What's Broken | Signal |
|---|---|---|---|
| `shape_mismatch` | 🟢 Easy | `nn.Linear` input dim wrong → explicit crash | Traceback |
| `training_collapse` | 🟡 Medium | Bad LR → NaN, or wrong loss fn → plateau | Loss curve |
| `wrong_device` | 🟡 Medium | Model on GPU, data on CPU → explicit crash | Traceback |
| `gradient_not_zeroed` | 🟠 Medium-Hard | Missing `zero_grad()` → gradients accumulate → NaN | Loss explosion |
| `data_leakage` | 🔴 Hard | Normalized before split → inflated metrics, no crash | Suspiciously high accuracy |
| `missing_eval_mode` | 🔴 Hard | No `model.eval()` → Dropout active → non-deterministic | Varying metrics |
| `compound_shape_device` | 🟠 Medium-Hard | **TWO bugs:** shape mismatch + device mismatch | Both must be fixed |
| `compound_leakage_eval` | 🟣 Expert | **TWO bugs:** data leakage + missing eval mode | Both silent, no crash |

The compound tasks are the hardest — two independent silent bugs in one script. Fix one and miss the other: 0.60. Fix both: 0.99.

---

## 🔧 Diagnostic Tools

The agent investigates using 5 tools, each costing 1 step:

| Tool | What It Does |
|---|---|
| `run_code` | Runs the buggy script in subprocess, returns stdout + stderr |
| `get_traceback` | Returns full Python traceback if code crashed |
| `inspect_gradients` | Injects gradient norm logging, runs one batch, returns per-layer norms |
| `print_shapes` | Injects forward hooks, returns tensor shapes at each layer |
| `view_source` | Reveals the full buggy source code |

The agent decides which tools to call and in what order. Calling `inspect_gradients` before `view_source` on a `gradient_not_zeroed` task is the optimal strategy — and the one the trained agent learns.

---

## 🏆 Scoring Ladder

```
0.01 → Wrong bug type identified
0.20 → Right type, fixed code crashes
0.40 → Code runs, training doesn't complete
0.60 → Training completes, root cause not fixed
0.80 → Root cause fixed, success signal missing
0.99 → Perfect fix ✅

+ Efficiency multiplier:
  Fix in ≤2 steps → ×1.2
  Fix in ≤3 steps → ×1.1
  Fix in 4-5 steps → ×1.0
  (capped at 0.99)
```

Plus an **LLM judge** (Groq / llama-3.3-70b) scores the agent's diagnosis on root cause correctness and mechanistic explanation — adding up to 0.15 reasoning reward on top of execution reward.

---

## 🧪 Adversarial Curriculum

An `AdversarialScheduler` tracks per-task performance across episodes. Bug types where the agent scores below 0.6 are marked "weak." Future `reset()` calls serve weak tasks 70% of the time with random seeds — novel code variants of the same bug pattern. The environment gets harder as the agent improves.

This mirrors the adversarial designer pattern from Kube SRE Gym — the training distribution adapts in real-time based on what the agent struggles with.

---

## 🤖 Training

**Model:** Qwen2.5-1.5B-Instruct + LoRA (4bit, rank 16) via Unsloth
**Method:** Custom GRPO loop
**Notebook:** [ml_debug_env_grpo.ipynb](https://github.com/RAK2315/ml-debug-env/blob/main/ml_debug_env_grpo.ipynb)

```
Run 1 (T4, 200 steps):  0.024 → 0.190  (+0.166)
Run 2 (H100, 500 steps): venue results — updated post-hackathon
```

The reward function is purely execution-based — the agent's fixed code is actually run in a subprocess. No regex matching, no string similarity. The code has to work.

---

## 📐 Action Space

Two action types:

**Inspect** (gather information, costs 1 step):
```json
{"action_type": "inspect", "tool_name": "run_code"}
```

**Fix** (submit complete fix, costs 1 step):
```json
{
  "action_type": "fix",
  "bug_type": "gradient_not_zeroed",
  "diagnosis": "optimizer.zero_grad() missing — gradients accumulate across batches causing explosion by epoch 2",
  "fixed_code": "import torch\n..."
}
```

Valid `bug_type` values: `shape_mismatch`, `training_collapse`, `data_leakage`, `wrong_device`, `gradient_not_zeroed`, `missing_eval_mode`, `compound_shape_device`, `compound_leakage_eval`, `other`

---

## 👁 Observation Space

| Field | When Set | Description |
|---|---|---|
| `task_id` | Always | Which task is active |
| `alert` | On reset | Minimal failure message — no code, no traceback |
| `available_tools` | Always | Tools the agent can call |
| `step_budget` | Always | Steps remaining |
| `num_bugs` | Always | 1 for single-bug tasks, 2 for compound tasks |
| `tool_name` | After inspect | Which tool was called |
| `tool_result` | After inspect | What the tool returned |
| `grader_score` | After fix | Score 0.01–0.99 |
| `grader_feedback` | After fix | Why this score was assigned |
| `efficiency_multiplier` | After fix | Bonus applied (1.0–1.2×) |
| `done` | Always | Episode over if score ≥ 0.95 or steps exhausted |

---

## 🚀 Quick Start

```python
import asyncio
from ml_debug_env import MlDebugEnvClient, DebugAction

async def main():
    async with MlDebugEnvClient(base_url="https://rak2315-ml-debug-env.hf.space") as env:
        # Reset — get only a minimal alert
        result = await env.reset()
        obs = result.observation
        print(f"Alert: {obs.alert}")
        print(f"Tools: {obs.available_tools}")
        print(f"Budget: {obs.step_budget} steps")

        # Inspect first — gather evidence
        result = await env.step(DebugAction(
            action_type="inspect",
            tool_name="run_code"
        ))
        print(f"Tool result: {result.observation.tool_result[:200]}")

        # Submit fix
        result = await env.step(DebugAction(
            action_type="fix",
            bug_type="shape_mismatch",
            diagnosis="Encoder outputs 256 but classifier expects 32",
            fixed_code="... complete fixed script ..."
        ))
        print(f"Score: {result.observation.grader_score}")
        print(f"Efficiency: {result.observation.efficiency_multiplier}x")

asyncio.run(main())
```

---

## 🌐 API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start episode — returns alert + tools only (no code) |
| `POST` | `/step` | inspect action or fix action |
| `GET` | `/state` | Episode state, tools used, best score |
| `GET` | `/health` | `{"status": "healthy"}` |
| `GET` | `/tasks` | All 8 tasks with descriptions and action schema |
| `POST` | `/grader` | Score a fix directly without a full episode |
| `GET` | `/baseline` | Run built-in LLM agent on all 8 tasks |
| `GET` | `/ui` | Landing page |
| `GET` | `/docs` | Interactive Swagger UI |
| `WS` | `/ws` | Persistent WebSocket session |

---

## 📁 File Structure

```
root/
├── inference.py               ← Validator entry point
├── models.py                  ← DebugAction, DebugObservation, DebugState
├── ml_debug_env_grpo.ipynb    ← GRPO training notebook (Unsloth + custom GRPO)
├── openenv.yaml               ← 8 tasks listed
├── Dockerfile                 ← HF Space deployment
└── server/
    ├── app.py                 ← FastAPI, all endpoints + /ui landing page
    ├── bug_generator.py       ← 8 tasks, 3 variants each, seeded + execute_tool()
    ├── grader.py              ← execution-based, AST checks, LLM judge
    ├── ml_debug_env_environment.py ← partial obs, tool handling, efficiency bonus
    ├── adversarial_scheduler.py    ← tracks weak tasks, skews resets
    ├── baseline_inference.py  ← Groq baseline agent
    └── landing_page.html      ← served at /ui
```

---

## 🔑 Environment Variables

| Variable | Set By | Description |
|---|---|---|
| `HF_TOKEN` | Injected by validator | Primary auth — checked first |
| `API_KEY` | Injected by validator | Fallback auth |
| `API_BASE_URL` | Injected by validator | LLM endpoint. Defaults to HF router. |
| `MODEL_NAME` | Injected by validator | Defaults to `Qwen/Qwen2.5-72B-Instruct` |
| `GROQ_API_KEY` | Developer | Local dev — enables `/baseline` |
| `PYTHON_EXEC` | Developer | Path to Python with torch for grader subprocess |

---

## 🛠 Local Development

```bash
git clone https://github.com/RAK2315/ml-debug-env
cd ml-debug-env
pip install -e .

# Windows
set GROQ_API_KEY=your_key
set PYTHON_EXEC=C:\path\to\python\with\torch\python.exe

uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl http://localhost:8000/tasks
```

---

## 📄 License

BSD 3-Clause — see [OpenEnv](https://github.com/meta-pytorch/OpenEnv/blob/main/LICENSE)
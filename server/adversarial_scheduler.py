"""
Block C — Adversarial Variant Generator

Tracks agent performance per bug type across episodes.
After 3+ attempts on a bug type, if mean score < 0.6, that type is "weak".
reset() skews toward weak types: 70% chance to pick a weak task if any exist.
Weak tasks also get random seeds to generate novel code variants.
This mirrors the curriculum + adversarial self-play pattern from Kube SRE Gym.
"""

import random
from collections import defaultdict
from typing import Optional, List, Dict


class AdversarialScheduler:
    """
    Tracks per-task scores and schedules harder variants of weak tasks.
    Thread-safe enough for single-agent use; for concurrent sessions,
    each environment instance gets its own scheduler.
    """

    WEAK_THRESHOLD = 0.6
    MIN_ATTEMPTS_BEFORE_WEAK = 3
    WEAK_TASK_PROB = 0.70

    def __init__(self, all_tasks: List[str]):
        self._all_tasks = all_tasks
        self._scores: Dict[str, List[float]] = defaultdict(list)
        self._episode_count = 0
        self._rng = random.Random()

    def record(self, task_id: str, score: float) -> None:
        """Call after each episode ends with the final score."""
        self._scores[task_id].append(score)

    def weak_tasks(self) -> List[str]:
        """Tasks with mean score < threshold after MIN_ATTEMPTS episodes."""
        weak = []
        for task_id, scores in self._scores.items():
            if len(scores) >= self.MIN_ATTEMPTS_BEFORE_WEAK:
                if (sum(scores) / len(scores)) < self.WEAK_THRESHOLD:
                    weak.append(task_id)
        return weak

    def next_task(self) -> str:
        """
        Pick next task. If weak tasks exist, pick one with WEAK_TASK_PROB probability.
        Otherwise round-robin through all tasks.
        """
        weak = self.weak_tasks()
        if weak and self._rng.random() < self.WEAK_TASK_PROB:
            task = self._rng.choice(weak)
        else:
            task = self._all_tasks[self._episode_count % len(self._all_tasks)]
        self._episode_count += 1
        return task

    def next_seed(self, task_id: str) -> int:
        """
        Weak tasks get random seeds → novel code variants.
        Strong tasks get deterministic seeds → consistent baseline.
        """
        weak = self.weak_tasks()
        if task_id in weak:
            return self._rng.randint(0, 99999)
        return 42

    def stats(self) -> Dict[str, dict]:
        """Return per-task performance stats for logging/debugging."""
        result = {}
        for task_id in self._all_tasks:
            scores = self._scores.get(task_id, [])
            result[task_id] = {
                "attempts": len(scores),
                "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
                "is_weak": task_id in self.weak_tasks(),
                "scores": scores[-5:],
            }
        return result
"""Curriculum difficulty scheduler for VLN-GRPO training.

Episodes are ranked by GT-trajectory length (number of actions). During early
training only short-trajectory episodes are allowed; the difficulty window
expands linearly until the full distribution is exposed by ramp_end_ratio.

This naturally corresponds to "start RL on short corridors, gradually increase
path complexity" without requiring a separate dataset split.

Integration:
    In ray_trainer.py, after computing `num_actions` for each episode, call
    `scheduler.should_include_episode(num_actions, global_step, total_steps)`.
    Episodes that return False are re-sampled from the remaining batch.
"""


class CurriculumScheduler:
    """Linear episode-difficulty curriculum over training steps.

    Args:
        min_actions: Maximum GT-action count allowed at the very start of training.
                     Episodes longer than this are excluded in the first phase.
        max_actions: Full-distribution cap; all episodes with ≤ max_actions
                     actions are allowed once the ramp is complete.
        ramp_start_ratio: Fraction of total_steps at which the ramp begins.
                          Before this point only min_actions episodes are used.
        ramp_end_ratio:   Fraction of total_steps at which the full distribution
                          is reached. After this point all episodes are allowed.
    """

    def __init__(
        self,
        min_actions: int = 10,
        max_actions: int = 200,
        ramp_start_ratio: float = 0.0,
        ramp_end_ratio: float = 0.70,
    ):
        self.min_actions = min_actions
        self.max_actions = max_actions
        self.ramp_start_ratio = ramp_start_ratio
        self.ramp_end_ratio = ramp_end_ratio

    def get_max_allowed_actions(self, step: int, total_steps: int) -> int:
        """Return the maximum GT-action count allowed at this training step."""
        if total_steps <= 0:
            return self.max_actions
        progress = step / total_steps
        if progress <= self.ramp_start_ratio:
            return self.min_actions
        if progress >= self.ramp_end_ratio:
            return self.max_actions
        t = (progress - self.ramp_start_ratio) / max(
            self.ramp_end_ratio - self.ramp_start_ratio, 1e-6
        )
        return int(round(self.min_actions + (self.max_actions - self.min_actions) * t))

    def should_include_episode(
        self, num_actions: int, step: int, total_steps: int
    ) -> bool:
        """True if an episode with *num_actions* GT actions is within the current
        difficulty window."""
        return num_actions <= self.get_max_allowed_actions(step, total_steps)

    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler(min={self.min_actions}, max={self.max_actions}, "
            f"ramp=[{self.ramp_start_ratio:.2f}, {self.ramp_end_ratio:.2f}])"
        )

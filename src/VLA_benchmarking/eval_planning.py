import dataclasses
import os

from .eval_io import (
    get_rollout_statuses,
    is_episode_complete,
    load_config,
    RolloutStatus,
)


@dataclasses.dataclass
class EpisodePlanEntry:
    """One episode's worth of planned work, whether reached via a standalone
    episode config or as one entry in an evaluation's episode list.
    """

    episode_config_path: str
    episode_config: dict
    episode_dir: str
    rollout_statuses: list[RolloutStatus]  # length num_rollouts
    is_complete: bool


@dataclasses.dataclass
class EvaluationPlan:
    """The full plan for one program invocation."""

    config_type: str  # "episode" or "evaluation"
    evaluation_name: "str | None"  # only set when config_type == "evaluation"
    episodes: "list[EpisodePlanEntry]"  # length 1 for a standalone episode


def _build_episode_plan_entry(episode_config_path: str, episode_config: dict, episode_dir: str) -> EpisodePlanEntry:
    """Compute an episode's current on-disk rollout status, given its
    already-loaded config, regardless of whether it's ever been run before.
    """
    rollout_statuses = get_rollout_statuses(episode_dir, filename="eval.yaml")
    if rollout_statuses is None:
        rollout_statuses = [RolloutStatus.NOT_FOUND] * episode_config["num_rollouts"]

    return EpisodePlanEntry(
        episode_config_path=episode_config_path,
        episode_config=episode_config,
        episode_dir=episode_dir,
        rollout_statuses=rollout_statuses,
        is_complete=is_episode_complete(rollout_statuses),
    )


def build_plan(config_path: str, policy: str, results_dir: str) -> EvaluationPlan:
    """Load config_path and build the full execution plan for this run."""
    config = load_config(config_path)

    if config["config_type"] == "episode":
        episode_dir = os.path.join(results_dir, policy, f"{config['task_name']}_episode0")
        entry = _build_episode_plan_entry(config_path, config, episode_dir)
        return EvaluationPlan(config_type="episode", evaluation_name=None, episodes=[entry])

    # config_type == "evaluation"
    evaluation_name = config["evaluation_name"]
    episode_paths = config["episode_paths"]

    occurrence_counts: dict = {}
    entries = []

    for episode_path in episode_paths:
        episode_num = occurrence_counts.get(episode_path, 0)
        occurrence_counts[episode_path] = episode_num + 1

        episode_config = load_config(episode_path)
        task_name = episode_config["task_name"]

        episode_dir = os.path.join(results_dir, policy, f"{task_name}_episode{episode_num}")
        entries.append(_build_episode_plan_entry(episode_path, episode_config, episode_dir))

    return EvaluationPlan(config_type="evaluation", evaluation_name=evaluation_name, episodes=entries)
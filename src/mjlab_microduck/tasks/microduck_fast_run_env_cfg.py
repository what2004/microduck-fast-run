"""Microduck fast-walking environment.

This task reuses the complete velocity environment and overrides only the
settings that differ for the first fast-walking training stage. The first
stage focuses on forward speed and holding a straight heading; lateral motion
and commanded turns can be introduced after the high-speed gait is stable.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)


def make_microduck_fast_run_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the first-stage straight fast-walking environment."""
    # Reuse the walking task's robot, 61D observations, rewards, BAM actuator,
    # domain randomization, events, terminations, and curricula.
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # Stage 1: learn a stable high-speed forward gait before adding lateral
    # motion and turning. Exact-zero standing samples are still supplied by
    # the inherited rel_standing_envs setting/curriculum.
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.30, 0.70)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_turn_in_place_envs = 0.0

    # Hold the spawn heading so high-speed yaw drift receives a corrective
    # angle-based signal. This must be removed or gated when commanded turns
    # are introduced in a later training stage.
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=1.0,
        params={
            "std": 0.4,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    return cfg


# Keep fast-run checkpoints and WandB runs separate from the base velocity task.
MicroduckFastRunRlCfg = deepcopy(MicroduckRlCfg)
MicroduckFastRunRlCfg.experiment_name = "fast_run"
MicroduckFastRunRlCfg.run_name = "fast_run"

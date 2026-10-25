# Copyright © MERL 2026
# Description: Adapts swarm observations and actions for MAPPO trainer interactions.
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import random
from typing import Any, Dict

import numpy as np
import torch

import central_critic as cc
import follower_actor as fa
import mappo_pipeline as mp
from rollout_visualization import plot_rollout_metrics, visualize_rollout_trajectory_3d
import swarm_env as se


def obs_adapter(env: Any, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert swarm_env next_state into MAPPOTrainer.collect_rollout structured inputs.

    Returned dict keys:
      - follower_actor_obs
      - critic_obs
      - meta (follower UAV ordering)
    """
    pos_dim = 3
    n_uavs = int(env.n_uavs)
    head_id = int(getattr(env, "head_uav_idx", 0))
    follower_ids = [uav_id for uav_id in range(n_uavs) if uav_id != head_id]
    follower_state = raw_obs["follower_state"]
    n_sensors = max(1, int(env.n_sensors))
    first_fstate = follower_state[follower_ids[0]] if follower_ids else {}
    range_scale = np.abs(np.max(env.space_span))
    u2s_range = float(
        first_fstate.get("u2s_range", env.u2s_range / range_scale)
    )
    u2u_range = float(
        first_fstate.get("u2u_range", env.u2u_range / range_scale)
    )

    follower_count = len(follower_ids)
    follower_sensor_pos = np.zeros(
        (1, follower_count, n_sensors, pos_dim), dtype=np.float32
    )
    follower_sensor_mask = np.zeros((1, follower_count, n_sensors), dtype=bool)
    follower_agent_pos = np.zeros((1, follower_count, pos_dim), dtype=np.float32)
    follower_agent_vel = np.zeros((1, follower_count, pos_dim), dtype=np.float32)
    follower_upstream_pos = np.zeros((1, follower_count, pos_dim), dtype=np.float32)
    follower_upstream_vel = np.zeros((1, follower_count, pos_dim), dtype=np.float32)
    follower_self_sensor_coverage = np.zeros((1, follower_count, 1), dtype=np.float32)
    follower_remaining_sensor_coverage = np.zeros((1, follower_count, 1), dtype=np.float32)
    follower_remaining_time = np.zeros((1, follower_count, 1), dtype=np.float32)
    for f_idx, uav_id in enumerate(follower_ids):
        fstate = follower_state[uav_id]
        sensor_pos = np.asarray(fstate["sensor_pos"], dtype=np.float32)
        if sensor_pos.size == 0:
            sensor_pos = np.empty((0, pos_dim), dtype=np.float32)
        else:
            sensor_pos = sensor_pos.reshape(-1, pos_dim)
        sensor_count = min(sensor_pos.shape[0], n_sensors)
        follower_sensor_pos[0, f_idx, :sensor_count] = sensor_pos[:sensor_count]
        follower_sensor_mask[0, f_idx, :sensor_count] = True

        self_pos = np.asarray(fstate["self_pos"], dtype=np.float32).reshape(-1, pos_dim)[0]
        self_vel = np.asarray(fstate["self_vel"], dtype=np.float32).reshape(-1, pos_dim)[0]
        upstream_pos_all = np.asarray(
            fstate["upstream_neighbor_pos"], dtype=np.float32
        )
        upstream_vel_all = np.asarray(
            fstate["upstream_neighbor_vel"], dtype=np.float32
        )
        upstream_pos = (
            upstream_pos_all.reshape(-1, pos_dim)[0]
            if upstream_pos_all.size == pos_dim
            else self_pos.copy()
        )
        upstream_vel = (
            upstream_vel_all.reshape(-1, pos_dim)[0]
            if upstream_vel_all.size == pos_dim
            else self_vel.copy()
        )

        follower_agent_pos[0, f_idx] = self_pos
        follower_agent_vel[0, f_idx] = self_vel
        follower_upstream_pos[0, f_idx] = upstream_pos
        follower_upstream_vel[0, f_idx] = upstream_vel
        follower_self_sensor_coverage[0, f_idx, 0] = fstate["self_sensor_coverage"]
        follower_remaining_sensor_coverage[0, f_idx, 0] = fstate["remaining_sensor_coverage"]
        follower_remaining_time[0, f_idx, 0] = fstate["remaining_time"]
    follower_mask = np.ones((1, follower_count), dtype=bool)

    follower_actor_obs = {
        "sensor_pos": torch.as_tensor(follower_sensor_pos, dtype=torch.float32),
        "agent_pos": torch.as_tensor(follower_agent_pos, dtype=torch.float32),
        "agent_vel": torch.as_tensor(follower_agent_vel, dtype=torch.float32),
        "self_sensor_coverage": torch.as_tensor(follower_self_sensor_coverage, dtype=torch.float32),
        "remaining_sensor_coverage": torch.as_tensor(follower_remaining_sensor_coverage, dtype=torch.float32),
        "remaining_time": torch.as_tensor(follower_remaining_time, dtype=torch.float32),
        "upstream_neighbor_pos": torch.as_tensor(follower_upstream_pos, dtype=torch.float32),
        "upstream_neighbor_vel": torch.as_tensor(follower_upstream_vel, dtype=torch.float32),
        "u2s_range": torch.as_tensor(u2s_range, dtype=torch.float32),
        "u2u_range": torch.as_tensor(u2u_range, dtype=torch.float32),
        "sensor_mask": torch.as_tensor(follower_sensor_mask, dtype=torch.bool),
        "follower_mask": torch.as_tensor(follower_mask, dtype=torch.bool),
    }

    critic_obs = {
        "follower_sensor_pos": torch.as_tensor(follower_sensor_pos, dtype=torch.float32),
        "follower_pos": torch.as_tensor(follower_agent_pos, dtype=torch.float32),
        "follower_vel": torch.as_tensor(follower_agent_vel, dtype=torch.float32),
        "follower_self_sensor_coverage": torch.as_tensor(follower_self_sensor_coverage, dtype=torch.float32),
        "follower_remaining_sensor_coverage": torch.as_tensor(
            follower_remaining_sensor_coverage, dtype=torch.float32
        ),
        "follower_remaining_time": torch.as_tensor(follower_remaining_time, dtype=torch.float32),
        "upstream_neighbor_pos": torch.as_tensor(follower_upstream_pos, dtype=torch.float32),
        "upstream_neighbor_vel": torch.as_tensor(follower_upstream_vel, dtype=torch.float32),
        "u2s_range": torch.as_tensor(u2s_range, dtype=torch.float32),
        "u2u_range": torch.as_tensor(u2u_range, dtype=torch.float32),
        "follower_sensor_mask": torch.as_tensor(follower_sensor_mask, dtype=torch.bool),
        "follower_mask": torch.as_tensor(follower_mask, dtype=torch.bool),
    }

    return {
        "follower_actor_obs": follower_actor_obs,
        "critic_obs": critic_obs,
        "meta": {
            "follower_uav_ids": follower_ids,
        },
    }


def action_adapter(
    follower_out: Dict[str, torch.Tensor],
    _structured_obs: Dict[str, Any],
) -> np.ndarray:
    """Convert follower direction actions into env.step actions; the environment fixes the head."""
    return follower_out["direction_action"][0].detach().cpu().numpy()


def reset_fn(env: Any) -> Dict[str, Any]:
    return env.reset_env(env.n_sensors)


def build_trainer(
    env, 
    rollout_length, 
    minibatch_size, 
    ppo_epochs, 
    direction_head_temperature: float = 1.0,
    use_mean_direction: bool = False,
) -> tuple[mp.MAPPOTrainer, Dict[str, Any]]:
    raw_obs = env.reset_env(env.n_sensors)
    structured = obs_adapter(env, raw_obs)
    position_dim = structured["follower_actor_obs"]["sensor_pos"].shape[-1]
    sensor_feature_dim = 2 * position_dim + 1
    upstream_feature_dim = 4 * position_dim + 2
    follower_actor = fa.Actor(
        sensor_feature_dim=sensor_feature_dim,
        upstream_feature_dim=upstream_feature_dim,
    )

    critic = cc.CentralCritic(
        follower_sensor_feature_dim=sensor_feature_dim,
        upstream_feature_dim=upstream_feature_dim,
    )

    trainer = mp.MAPPOTrainer(
        follower_actor=follower_actor,
        critic=critic,
        config=mp.MAPPOConfig(
            rollout_length=rollout_length,
            minibatch_size=minibatch_size,
            ppo_epochs=ppo_epochs,
            direction_temperature=direction_head_temperature,
            use_mean_direction=use_mean_direction,
        ),
        device=torch.device("cpu"),
    )

    return trainer, raw_obs

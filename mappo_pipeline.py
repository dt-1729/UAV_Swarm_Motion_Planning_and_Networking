# Copyright © MERL 2026
# Description: Implements MAPPO rollout collection, advantage estimation, and policy updates.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from rollout_buffer import MAPPORolloutBuffer


NestedData = Dict[str, Any]


def _scheduled_temperature(
    initial: float,
    final: float,
    step: int,
    total_steps: int,
    schedule: str,
) -> float:
    """Interpolate between `initial` and `final` temperatures over `total_steps`.

    Args:
        initial: Starting temperature (step 1).
        final:   Ending temperature (step total_steps).
        step:    Current 1-based update index.
        total_steps: Total number of training updates.
        schedule: One of "linear", "cosine", or "exponential".

    Returns:
        Scheduled temperature for this step.
    """
    if total_steps <= 1:
        return initial
    # Progress in [0, 1]; 0 at first step, 1 at last.
    t = (step - 1) / (total_steps - 1)
    if schedule == "linear":
        return initial + t * (final - initial)
    elif schedule == "cosine":
        return final + 0.5 * (initial - final) * (1.0 + math.cos(math.pi * t))
    elif schedule == "exponential":
        # Avoid log(0); clamp final to a small positive value.
        safe_final = max(final, 1e-8)
        return initial * (safe_final / initial) ** t
    else:
        raise ValueError(f"Unknown temperature schedule: {schedule!r}. Choose linear, cosine, or exponential.")



@dataclass
class MAPPOConfig:
    rollout_length: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 8
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip_norm: float = 0.5
    follower_actor_lr: float = 7e-4
    critic_lr: float = 5e-4
    direction_temperature: float = 1.0
    normalize_advantages: bool = True
    stop_on_termination: bool = False
    rollout_progress: bool = False
    rollout_progress_every: int = 1
    record_rollout_trajectory: bool = False
    use_mean_direction: bool = False


class MAPPOTrainer:
    """
    MAPPO trainer with:
    - a shared follower actor
    - centralized critic
    - on-policy rollout collection
    - PPO clipped policy/value updates

    Observation contract expected from obs_adapter:
    {
      "follower_actor_obs": {
          "sensor_pos": (B,F,N,d),
          "agent_pos": (B,F,d),
          "agent_vel": (B,F,d),
          "self_sensor_coverage": (B,F,1),
          "remaining_sensor_coverage": (B,F,1),
          "remaining_time": (B,F,1),
          "upstream_neighbor_pos": (B,F,d),
          "upstream_neighbor_vel": (B,F,d),
          "u2s_range": scalar or (B,F) or (B,F,1),
          "u2u_range": scalar or (B,F) or (B,F,1),
          optional "sensor_mask": (B,F,N),
          optional "follower_mask": (B,F),
      },
      "critic_obs": {kwargs for central_critic.forward}
    }
    """

    def __init__(
        self,
        follower_actor: nn.Module,
        critic: nn.Module,
        config: MAPPOConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.follower_actor = follower_actor
        self.critic = critic
        self.config = config

        self.device = device if device is not None else torch.device("cpu")
        self.follower_actor.to(self.device)
        self.critic.to(self.device)

        self.buffer = MAPPORolloutBuffer(
            rollout_length=config.rollout_length,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        self.follower_actor_optimizer = torch.optim.Adam(self.follower_actor.parameters(), lr=config.follower_actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    def _check_required_actor_api(self) -> None:
        if not hasattr(self.follower_actor, "act"):
            raise RuntimeError("follower_actor must implement act(...)")
        if not hasattr(self.follower_actor, "evaluate_actions"):
            raise RuntimeError("follower_actor must implement evaluate_actions(...)")

    @staticmethod
    def _to_device(data: Any, device: torch.device) -> Any:
        if isinstance(data, dict):
            return {k: MAPPOTrainer._to_device(v, device) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(MAPPOTrainer._to_device(v, device) for v in data)
        if isinstance(data, torch.Tensor):
            return data.to(device)
        return torch.as_tensor(data, device=device)

    @staticmethod
    def _expand_scalar_to_batch(value: Any, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        t = t.to(device=device, dtype=dtype)

        if t.ndim == 0:
            t = t.view(1, 1).expand(batch_size, 1)
        elif t.ndim == 1:
            if t.shape[0] == 1:
                t = t.view(1, 1).expand(batch_size, 1)
            elif t.shape[0] == batch_size:
                t = t.view(batch_size, 1)
            elif batch_size % t.shape[0] == 0:
                repeat_factor = batch_size // t.shape[0]
                t = t.repeat_interleave(repeat_factor).view(batch_size, 1)
            else:
                raise ValueError(f"Cannot expand 1D tensor with shape {tuple(t.shape)} to batch {batch_size}")
        elif t.ndim == 2 and t.shape[1] == 1:
            if t.shape[0] == batch_size:
                pass
            elif batch_size % t.shape[0] == 0:
                repeat_factor = batch_size // t.shape[0]
                t = t.repeat_interleave(repeat_factor, dim=0)
            else:
                raise ValueError(f"Cannot expand 2D tensor with shape {tuple(t.shape)} to batch {batch_size}")
        else:
            raise ValueError(f"Expected scalar-like range input, got shape {tuple(t.shape)}")
        return t

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=dim)
        w = mask.float()
        denom = w.sum(dim=dim).clamp_min(1.0)
        return (x * w).sum(dim=dim) / denom

    @staticmethod
    def _masked_sum(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
        if mask is None:
            return x.sum(dim=dim)
        return (x * mask.float()).sum(dim=dim)

    @staticmethod
    def _ensure_batch_vector(x: Any, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.view(1).expand(batch_size)
        elif t.ndim == 1 and t.shape[0] == batch_size:
            pass
        elif t.ndim == 1 and t.shape[0] == 1:
            t = t.expand(batch_size)
        else:
            raise ValueError(f"Expected shape (B,) or scalar for reward/done/value; got {tuple(t.shape)}")
        return t

    # def _forward_predefined_head_act(
    #     self,
    #     env: Any,
    #     structured_obs: NestedData,
    #     head_actor_obs: NestedData,
    # ) -> Dict[str, torch.Tensor]:
    #     head_id = int(getattr(env, "head_uav_idx", 0))

    #     meta = structured_obs.get("meta", {})
    #     head_sensor_ids_raw = meta.get("head_sensor_ids", list(range(int(getattr(env, "n_sensors", 0)))))
    #     head_sensor_ids = [self._to_int(x) for x in head_sensor_ids_raw]

    #     if len(head_sensor_ids) == 0:
    #         raise ValueError("head_sensor_ids is empty; cannot build predefined head action.")

    #     visited = np.asarray(getattr(env, "visited", np.zeros(int(getattr(env, "n_sensors", 0)), dtype=bool)), dtype=bool)
    #     visit_sequence = np.asarray(
    #         getattr(env, "sensor_visit_sequence", np.arange(int(getattr(env, "n_sensors", 0)))),
    #         dtype=np.int64,
    #     ).tolist()

    #     head_pos = np.asarray(env.uav_locs[head_id], dtype=np.float32)
    #     u2s_range = float(getattr(env, "u2s_range", 0.0))

    #     target_sensor = None
    #     for sensor_idx in visit_sequence:
    #         sensor_idx = int(sensor_idx)
    #         if 0 <= sensor_idx < visited.shape[0] and not visited[sensor_idx]:
    #             # Skip sensors the head has already arrived at so it keeps advancing
    #             # along the sequence rather than hovering at the first waypoint.
    #             dist = float(np.linalg.norm(head_pos - np.asarray(env.sensor_locs[sensor_idx], dtype=np.float32)))
    #             if dist <= u2s_range:
    #                 continue
    #             target_sensor = sensor_idx
    #             break

    #     if target_sensor is None:
    #         for sensor_idx in head_sensor_ids:
    #             if 0 <= sensor_idx < visited.shape[0] and not visited[sensor_idx]:
    #                 dist = float(np.linalg.norm(head_pos - np.asarray(env.sensor_locs[sensor_idx], dtype=np.float32)))
    #                 if dist <= u2s_range:
    #                     continue
    #                 target_sensor = sensor_idx
    #                 break

    #     # Final fallback: any unvisited sensor regardless of proximity.
    #     if target_sensor is None:
    #         for sensor_idx in range(visited.shape[0]):
    #             if not visited[sensor_idx]:
    #                 target_sensor = sensor_idx
    #                 break

    #     if target_sensor is None:
    #         target_sensor = int(head_sensor_ids[0])
 
    #     if target_sensor in head_sensor_ids:
    #         local_sensor_action = head_sensor_ids.index(target_sensor)
    #     else:
    #         local_sensor_action = 0
    #         target_sensor = int(head_sensor_ids[0])
    #     target_pos = np.asarray(env.sensor_locs[target_sensor], dtype=np.float32)
    #     direction_vec = target_pos - head_pos
    #     norm = float(np.linalg.norm(direction_vec))
    #     if norm > 1e-8:
    #         direction_vec = direction_vec / norm
    #     else:
    #         direction_vec = np.zeros(3, dtype=np.float32)

    #     batch_size = int(head_actor_obs["sensor_pos"].shape[0])
    #     sensor_action = torch.full((batch_size,), int(local_sensor_action), dtype=torch.long, device=self.device)
    #     direction_action = torch.as_tensor(direction_vec, dtype=torch.float32, device=self.device).view(1, 3).expand(batch_size, -1).clone()

    #     with torch.no_grad():
    #         head_eval = self.head_actor.evaluate_actions(
    #             **head_actor_obs,
    #             sensor_action=sensor_action,
    #             direction_action=direction_action,
    #             sensor_temperature=self.config.sensor_temperature,
    #             direction_temperature=self.config.direction_temperature,
    #         )

    #     return {
    #         "sensor_action": sensor_action,
    #         "direction_action": direction_action,
    #         "direction_weights": direction_action,
    #         "log_prob": head_eval["log_prob"],
    #     }

    # def _forward_stationary_head_act(
    #     self,
    #     env: Any,
    #     structured_obs: NestedData,
    #     head_actor_obs: NestedData,
    # ) -> Dict[str, torch.Tensor]:
    #     head_id = int(getattr(env, "head_uav_idx", 0))

    #     meta = structured_obs.get("meta", {})
    #     head_sensor_ids_raw = meta.get("head_sensor_ids", list(range(int(getattr(env, "n_sensors", 0)))))
    #     head_sensor_ids = [self._to_int(x) for x in head_sensor_ids_raw]
    #     if len(head_sensor_ids) == 0:
    #         raise ValueError("head_sensor_ids is empty; cannot build stationary head action.")

    #     visited = np.asarray(getattr(env, "visited", np.zeros(int(getattr(env, "n_sensors", 0)), dtype=bool)), dtype=bool)
    #     target_sensor = None
    #     for sensor_idx in head_sensor_ids:
    #         if 0 <= sensor_idx < visited.shape[0] and not visited[sensor_idx]:
    #             target_sensor = int(sensor_idx)
    #             break
    #     if target_sensor is None:
    #         target_sensor = int(head_sensor_ids[0])

    #     local_sensor_action = head_sensor_ids.index(target_sensor) if target_sensor in head_sensor_ids else 0

    #     head_pos = np.asarray(env.uav_locs[head_id], dtype=np.float32)
    #     target_pos = np.asarray(env.sensor_locs[target_sensor], dtype=np.float32)
    #     direction_vec = target_pos - head_pos
    #     norm = float(np.linalg.norm(direction_vec))
    #     if norm > 1e-8:
    #         direction_vec = direction_vec / norm
    #     else:
    #         direction_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    #     batch_size = int(head_actor_obs["sensor_pos"].shape[0])
    #     sensor_action = torch.full((batch_size,), int(local_sensor_action), dtype=torch.long, device=self.device)
    #     direction_action = (
    #         torch.as_tensor(direction_vec, dtype=torch.float32, device=self.device)
    #         .view(1, 3)
    #         .expand(batch_size, -1)
    #         .clone()
    #     )

    #     with torch.no_grad():
    #         head_eval = self.head_actor.evaluate_actions(
    #             **head_actor_obs,
    #             sensor_action=sensor_action,
    #             direction_action=direction_action,
    #             sensor_temperature=self.config.sensor_temperature,
    #             direction_temperature=self.config.direction_temperature,
    #         )

    #     return {
    #         "sensor_action": sensor_action,
    #         "direction_action": direction_action,
    #         "direction_weights": direction_action,
    #         "log_prob": head_eval["log_prob"],
    #     }

    # def _forward_diagonal_head_act(
    #     self,
    #     env: Any,
    #     structured_obs: NestedData,
    #     head_actor_obs: NestedData,
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     Move the head UAV back-and-forth along the field diagonal
    #     (space_min XY corner ↔ space_max XY corner) at its current altitude.

    #     A lightweight target corner is persisted on the env object as
    #     ``env._head_diagonal_target`` (0 = space_min corner, 1 = space_max corner)
    #     so the head advances all the way to each corner before reversing.
    #     """
    #     head_id = int(getattr(env, "head_uav_idx", 0))
    #     head_pos = np.asarray(env.uav_locs[head_id], dtype=np.float32)

    #     space_min = np.asarray(env.space_min, dtype=np.float32)
    #     space_max = np.asarray(env.space_max, dtype=np.float32)
    #     # Keep z fixed at the head's current altitude.
    #     corner_a = np.array([space_min[0], space_min[1], head_pos[2]], dtype=np.float32)
    #     corner_b = np.array([space_max[0], space_max[1], head_pos[2]], dtype=np.float32)

    #     # Initialise or retrieve which corner is the current destination.
    #     if not hasattr(env, "_head_diagonal_target"):
    #         # Start by heading toward whichever corner is farther away.
    #         env._head_diagonal_target = int(
    #             np.linalg.norm(head_pos - corner_a) < np.linalg.norm(head_pos - corner_b)
    #         )  # 0 → corner_a is near, head toward corner_b; 1 → head toward corner_a

    #     target_corner = corner_b if env._head_diagonal_target == 1 else corner_a
    #     arrival_threshold = float(getattr(env, "u2u_range", 0.0)) * 0.5
    #     arrival_threshold = max(arrival_threshold, float(np.linalg.norm(space_max - space_min)) * 0.05)

    #     dist_to_target = float(np.linalg.norm(head_pos - target_corner))
    #     if dist_to_target <= arrival_threshold:
    #         # Arrived — flip destination.
    #         env._head_diagonal_target = 1 - env._head_diagonal_target
    #         target_corner = corner_b if env._head_diagonal_target == 1 else corner_a

    #     direction_vec = target_corner - head_pos
    #     norm = float(np.linalg.norm(direction_vec))
    #     direction_vec = direction_vec / norm if norm > 1e-8 else np.array([1.0, 1.0, 0.0], dtype=np.float32) / np.sqrt(2)

    #     # Sensor action: pick the first unvisited sensor in head's visible set.
    #     meta = structured_obs.get("meta", {})
    #     head_sensor_ids_raw = meta.get("head_sensor_ids", list(range(int(getattr(env, "n_sensors", 0)))))
    #     head_sensor_ids = [self._to_int(x) for x in head_sensor_ids_raw]
    #     if not head_sensor_ids:
    #         head_sensor_ids = [0]
    #     visited = np.asarray(getattr(env, "visited", np.zeros(int(getattr(env, "n_sensors", 0)), dtype=bool)), dtype=bool)
    #     target_sensor = next(
    #         (s for s in head_sensor_ids if 0 <= s < visited.shape[0] and not visited[s]),
    #         int(head_sensor_ids[0]),
    #     )
    #     local_sensor_action = head_sensor_ids.index(target_sensor) if target_sensor in head_sensor_ids else 0

    #     batch_size = int(head_actor_obs["sensor_pos"].shape[0])
    #     sensor_action = torch.full((batch_size,), int(local_sensor_action), dtype=torch.long, device=self.device)
    #     direction_action = (
    #         torch.as_tensor(direction_vec, dtype=torch.float32, device=self.device)
    #         .view(1, 3)
    #         .expand(batch_size, -1)
    #         .clone()
    #     )

    #     with torch.no_grad():
    #         head_eval = self.head_actor.evaluate_actions(
    #             **head_actor_obs,
    #             sensor_action=sensor_action,
    #             direction_action=direction_action,
    #             sensor_temperature=self.config.sensor_temperature,
    #             direction_temperature=self.config.direction_temperature,
    #         )

    #     return {
    #         "sensor_action": sensor_action,
    #         "direction_action": direction_action,
    #         "direction_weights": direction_action,
    #         "log_prob": head_eval["log_prob"],
    #     }

    # def _forward_perimeter_head_act(
    #     self,
    #     env: Any,
    #     structured_obs: NestedData,
    #     head_actor_obs: NestedData,
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     Move the head UAV clockwise along the square boundary in XY
    #     (bottom edge -> right edge -> top edge -> left edge), keeping z fixed.
    #     """
    #     head_id = int(getattr(env, "head_uav_idx", 0))
    #     head_pos = np.asarray(env.uav_locs[head_id], dtype=np.float32)

    #     xmin, ymin = float(env.space_min[0]), float(env.space_min[1])
    #     xmax, ymax = float(env.space_max[0]), float(env.space_max[1])
    #     z = float(head_pos[2])

    #     corners = [
    #         np.array([xmin, ymin, z], dtype=np.float32),
    #         np.array([xmax, ymin, z], dtype=np.float32),
    #         np.array([xmax, ymax, z], dtype=np.float32),
    #         np.array([xmin, ymax, z], dtype=np.float32),
    #     ]

    #     if not hasattr(env, "_head_perimeter_target_idx"):
    #         nearest_corner_idx = int(np.argmin([np.linalg.norm(head_pos - c) for c in corners]))
    #         env._head_perimeter_target_idx = int((nearest_corner_idx + 1) % 4)

    #     target_idx = int(getattr(env, "_head_perimeter_target_idx", 1)) % 4
    #     target_corner = corners[target_idx]

    #     field_span_xy = max(float(xmax - xmin), float(ymax - ymin))
    #     arrival_threshold = max(float(getattr(env, "u2u_range", 0.0)) * 0.25, 0.01 * field_span_xy)

    #     dist_to_target = float(np.linalg.norm(head_pos - target_corner))
    #     if dist_to_target <= arrival_threshold:
    #         target_idx = (target_idx + 1) % 4
    #         env._head_perimeter_target_idx = target_idx
    #         target_corner = corners[target_idx]

    #     direction_vec = target_corner - head_pos
    #     norm = float(np.linalg.norm(direction_vec))
    #     direction_vec = direction_vec / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

    #     # Sensor action: pick the first unvisited sensor in head's visible set.
    #     meta = structured_obs.get("meta", {})
    #     head_sensor_ids_raw = meta.get("head_sensor_ids", list(range(int(getattr(env, "n_sensors", 0)))))
    #     head_sensor_ids = [self._to_int(x) for x in head_sensor_ids_raw]
    #     if not head_sensor_ids:
    #         head_sensor_ids = [0]
    #     visited = np.asarray(getattr(env, "visited", np.zeros(int(getattr(env, "n_sensors", 0)), dtype=bool)), dtype=bool)
    #     target_sensor = next(
    #         (s for s in head_sensor_ids if 0 <= s < visited.shape[0] and not visited[s]),
    #         int(head_sensor_ids[0]),
    #     )
    #     local_sensor_action = head_sensor_ids.index(target_sensor) if target_sensor in head_sensor_ids else 0

    #     batch_size = int(head_actor_obs["sensor_pos"].shape[0])
    #     sensor_action = torch.full((batch_size,), int(local_sensor_action), dtype=torch.long, device=self.device)
    #     direction_action = (
    #         torch.as_tensor(direction_vec, dtype=torch.float32, device=self.device)
    #         .view(1, 3)
    #         .expand(batch_size, -1)
    #         .clone()
    #     )

    #     with torch.no_grad():
    #         head_eval = self.head_actor.evaluate_actions(
    #             **head_actor_obs,
    #             sensor_action=sensor_action,
    #             direction_action=direction_action,
    #             sensor_temperature=self.config.sensor_temperature,
    #             direction_temperature=self.config.direction_temperature,
    #         )

    #     return {
    #         "sensor_action": sensor_action,
    #         "direction_action": direction_action,
    #         "direction_weights": direction_action,
    #         "log_prob": head_eval["log_prob"],
    #     }

    def _forward_follower_act(self, follower_obs: NestedData) -> Dict[str, torch.Tensor]:
        sensor_pos = follower_obs["sensor_pos"]
        agent_pos = follower_obs["agent_pos"]
        agent_vel = follower_obs["agent_vel"]
        upstream_neighbor_pos = follower_obs["upstream_neighbor_pos"]
        upstream_neighbor_vel = follower_obs["upstream_neighbor_vel"]
        sensor_mask = follower_obs["sensor_mask"]

        b, f, n, d = sensor_pos.shape
        bf = b * f

        if agent_vel is None:
            agent_vel = torch.zeros_like(agent_pos)
        if upstream_neighbor_vel is None:
            upstream_neighbor_vel = torch.zeros_like(upstream_neighbor_pos)

        flat_sensor_pos = sensor_pos.reshape(bf, n, d)
        flat_agent_pos = agent_pos.reshape(bf, d)
        flat_agent_vel = agent_vel.reshape(bf, d)
        flat_upstream = upstream_neighbor_pos.reshape(bf, d)
        flat_upstream_vel = upstream_neighbor_vel.reshape(bf, d)
        flat_sensor_mask = (
            None if sensor_mask is None else sensor_mask[..., :n].reshape(bf, n)
        )

        flat_u2s = self._expand_scalar_to_batch(
            follower_obs["u2s_range"], bf, flat_agent_pos.device, flat_agent_pos.dtype
        )
        flat_u2u = self._expand_scalar_to_batch(
            follower_obs["u2u_range"], bf, flat_agent_pos.device, flat_agent_pos.dtype
        )

        act_out = self.follower_actor.act(
            sensor_pos=flat_sensor_pos,
            agent_pos=flat_agent_pos,
            agent_vel=flat_agent_vel,
            self_sensor_coverage=follower_obs["self_sensor_coverage"].reshape(bf, 1),
            remaining_sensor_coverage=follower_obs["remaining_sensor_coverage"].reshape(bf, 1),
            remaining_time=follower_obs["remaining_time"].reshape(bf, 1),
            upstream_neighbor_pos=flat_upstream,
            upstream_neighbor_vel=flat_upstream_vel,
            u2s_range=flat_u2s,
            u2u_range=flat_u2u,
            sensor_mask=flat_sensor_mask,
            direction_temperature=self.config.direction_temperature,
            use_mean_direction=self.config.use_mean_direction,
        )

        return {
            "direction_action": act_out["direction_action"].reshape(b, f, d),
            "log_prob": act_out["log_prob"].reshape(b, f),
        }

    def _forward_follower_eval(
        self,
        follower_obs: NestedData,
        direction_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sensor_pos = follower_obs["sensor_pos"]
        agent_pos = follower_obs["agent_pos"]
        agent_vel = follower_obs["agent_vel"]
        self_sensor_coverage = follower_obs["self_sensor_coverage"]
        remaining_sensor_coverage = follower_obs["remaining_sensor_coverage"]
        remaining_time = follower_obs["remaining_time"]
        upstream_neighbor_pos = follower_obs["upstream_neighbor_pos"]
        upstream_neighbor_vel = follower_obs["upstream_neighbor_vel"]
        sensor_mask = follower_obs["sensor_mask"]

        b, f, n, d = sensor_pos.shape
        bf = b * f

        if agent_vel is None:
            agent_vel = torch.zeros_like(agent_pos)
        if upstream_neighbor_vel is None:
            upstream_neighbor_vel = torch.zeros_like(upstream_neighbor_pos)

        flat_sensor_pos = sensor_pos.reshape(bf, n, d)
        flat_agent_pos = agent_pos.reshape(bf, d)
        flat_agent_vel = agent_vel.reshape(bf, d)
        flat_self_sensor_coverage = self_sensor_coverage.reshape(bf, 1)
        flat_remaining_sensor_coverage = remaining_sensor_coverage.reshape(bf, 1)
        flat_remaining_time = remaining_time.reshape(bf, 1)
        flat_upstream = upstream_neighbor_pos.reshape(bf, d)
        flat_upstream_vel = upstream_neighbor_vel.reshape(bf, d)
        flat_sensor_mask = (
            None if sensor_mask is None else sensor_mask[..., :n].reshape(bf, n)
        )

        flat_direction_action = direction_action.reshape(bf, d)

        flat_u2s = self._expand_scalar_to_batch(
            follower_obs["u2s_range"], bf, flat_agent_pos.device, flat_agent_pos.dtype
        )
        flat_u2u = self._expand_scalar_to_batch(
            follower_obs["u2u_range"], bf, flat_agent_pos.device, flat_agent_pos.dtype
        )

        eval_out = self.follower_actor.evaluate_actions(
            sensor_pos=flat_sensor_pos,
            agent_pos=flat_agent_pos,
            agent_vel=flat_agent_vel,
            self_sensor_coverage=flat_self_sensor_coverage,
            remaining_sensor_coverage=flat_remaining_sensor_coverage,
            remaining_time=flat_remaining_time,
            direction_action=flat_direction_action,
            upstream_neighbor_pos=flat_upstream,
            upstream_neighbor_vel=flat_upstream_vel,
            u2s_range=flat_u2s,
            u2u_range=flat_u2u,
            sensor_mask=flat_sensor_mask,
            direction_temperature=self.config.direction_temperature,
        )

        flat_log_prob = eval_out["log_prob"]
        flat_entropy = eval_out["entropy"]

        return flat_log_prob.reshape(b, f), flat_entropy.reshape(b, f)

    def collect_rollout(
        self,
        env: Any,
        obs_adapter: Callable[[Any, Any], NestedData],
        action_adapter: Callable[[Dict[str, torch.Tensor], NestedData], Any],
        reset_fn: Optional[Callable[[Any], Any]] = None,
        initial_obs: Optional[Any] = None,
        stop_on_termination: Optional[bool] = None,
        rollout_progress: Optional[bool] = None,
        rollout_progress_every: Optional[int] = None,
        record_rollout_trajectory: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Collect one rollout of length config.rollout_length.

        obs_adapter(env, raw_obs) -> structured obs dict with keys:
            follower_actor_obs, critic_obs

        action_adapter(follower_act_out, structured_obs) -> env action
        """
        self._check_required_actor_api()
        stop_on_termination = self.config.stop_on_termination if stop_on_termination is None else stop_on_termination
        rollout_progress = self.config.rollout_progress if rollout_progress is None else rollout_progress
        rollout_progress_every = (
            self.config.rollout_progress_every if rollout_progress_every is None else rollout_progress_every
        )
        rollout_progress_every = max(1, int(rollout_progress_every))
        record_rollout_trajectory = (
            self.config.record_rollout_trajectory
            if record_rollout_trajectory is None
            else record_rollout_trajectory
        )
        if initial_obs is None:
            if reset_fn is not None:
                raw_obs = reset_fn(env)
            else:
                raw_obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        else:
            raw_obs = initial_obs

        self.buffer.reset()
        total_reward = 0.0
        total_abs_reward = 0.0
        total_terminated = 0.0
        total_truncated = 0.0
        nonzero_reward_steps = 0
        total_collision_penalty = 0.0
        total_connectivity_penalty = 0.0
        total_sensor_potential_reward = 0.0
        total_sensor_collection_reward = 0.0
        steps_collected = 0
        ended_on_termination = False
        last_visited_fraction = float(np.mean(np.asarray(env.visited, dtype=np.float32)))
        episode_end_visited_fractions: list[float] = []

        trajectory_record: Optional[Dict[str, Any]] = None
        if record_rollout_trajectory:
            trajectory_record = {
                "uav_locs": [np.asarray(env.uav_locs, dtype=np.float32).copy()],
                "times": [float(getattr(env, "current_time", 0.0))],
                "reward_per_step": [],
                "visited_fraction": [float(np.mean(np.asarray(env.visited, dtype=np.float32)))],
                "visited_mask": [np.asarray(env.visited, dtype=bool).copy()],
                "connectivity": [np.asarray(getattr(env, "uav_connectivity", np.zeros((env.n_uavs, env.n_uavs), dtype=bool)), dtype=bool).copy()],
                "sensor_locs": np.asarray(env.sensor_locs, dtype=np.float32).copy(),
                "base_station": np.asarray(env.bs_loc, dtype=np.float32).copy(),
                "u2u_range": float(env.u2u_range),
                "u2u_safe_dist": float(env.u2u_safe_dist),
                "space_min": np.asarray(env.space_min, dtype=np.float32).copy(),
                "space_max": np.asarray(env.space_max, dtype=np.float32).copy(),
                "head_uav_idx": int(getattr(env, "head_uav_idx", 0)),
                "follower_uav_ids": [],
            }

        if rollout_progress:
            print(
                f"[collect_rollout] start: target_steps={self.config.rollout_length}, "
                f"stop_on_termination={stop_on_termination}"
            )

        for t in range(self.config.rollout_length):
            structured_obs = obs_adapter(env, raw_obs)
            structured_obs = self._to_device(structured_obs, self.device)

            follower_actor_obs = structured_obs["follower_actor_obs"]
            critic_obs = structured_obs["critic_obs"]

            with torch.no_grad():
                follower_out = self._forward_follower_act(follower_actor_obs)
                values = self.critic(**critic_obs)

            env_action = action_adapter(follower_out, structured_obs)
            next_raw_obs, reward, terminated, truncated, info = env.step(env_action)

            batch_size = int(values.shape[0]) if values.ndim == 1 else int(values.shape[0])
            rewards_t = self._ensure_batch_vector(reward, batch_size, self.device, torch.float32)
            terminated_t = self._ensure_batch_vector(terminated, batch_size, self.device, torch.float32).bool()
            truncated_t = self._ensure_batch_vector(truncated, batch_size, self.device, torch.float32).bool()

            # also truncate at rollout boundary
            if t == self.config.rollout_length - 1:
                truncated_t = torch.ones_like(terminated_t)

            bootstrap_values = None
            boundary_t = terminated_t | truncated_t
            if bool(torch.any(boundary_t).item()):
                with torch.no_grad():
                    next_structured_obs = self._to_device(obs_adapter(env, next_raw_obs), self.device)
                    bootstrap_values = self.critic(**next_structured_obs["critic_obs"])
                    bootstrap_values = torch.where(
                        terminated_t,
                        torch.zeros_like(bootstrap_values),
                        bootstrap_values,
                    )

            self.buffer.add(
                follower_obs=follower_actor_obs,
                critic_obs=critic_obs,
                follower_direction_actions=follower_out["direction_action"],
                follower_log_probs=follower_out["log_prob"],
                rewards=rewards_t,
                terminated=terminated_t,
                truncated=truncated_t,
                values=values,
                bootstrap_values=bootstrap_values,
                extras={
                    "info": info,
                },
            )

            total_reward += float(rewards_t.mean().item())
            total_abs_reward += float(torch.abs(rewards_t).mean().item())
            total_terminated += float(terminated_t.float().mean().item())
            total_truncated += float(truncated_t.float().mean().item())
            steps_collected += 1

            step_reward_mean = float(rewards_t.mean().item())
            if abs(step_reward_mean) > 1e-12:
                nonzero_reward_steps += 1

            if isinstance(info, dict):
                total_collision_penalty += float(info.get("reward_collision_penalty", 0.0))
                total_connectivity_penalty += float(info.get("reward_connectivity_penalty", 0.0))
                total_sensor_potential_reward += float(info.get("reward_sensor_potential", 0.0))
                total_sensor_collection_reward += float(info.get("reward_sensor_collection", 0.0))

            visited_fraction_t = float(np.mean(np.asarray(env.visited, dtype=np.float32)))
            last_visited_fraction = visited_fraction_t

            if rollout_progress and (
                steps_collected == 1
                or steps_collected % rollout_progress_every == 0
                or steps_collected == self.config.rollout_length
            ):
                print(
                    f"[collect_rollout] step={steps_collected}/{self.config.rollout_length} "
                    f"reward_mean={step_reward_mean:.4f} "
                    f"cumulative_reward={total_reward:.4f} "
                    f"cumulative_abs_reward={total_abs_reward:.4f} "
                    f"terminated={float(terminated_t.float().mean().item()):.4f} "
                    f"truncated={float(truncated_t.float().mean().item()):.4f} "
                    f"visited_fraction={visited_fraction_t:.4f}"
                )

            if trajectory_record is not None:
                trajectory_record["uav_locs"].append(np.asarray(env.uav_locs, dtype=np.float32).copy())
                trajectory_record["times"].append(float(getattr(env, "current_time", 0.0)))
                trajectory_record["reward_per_step"].append(float(rewards_t.mean().item()))
                trajectory_record["visited_fraction"].append(visited_fraction_t)
                trajectory_record["visited_mask"].append(np.asarray(env.visited, dtype=bool).copy())
                trajectory_record["connectivity"].append(
                    np.asarray(
                        getattr(env, "uav_connectivity", np.zeros((env.n_uavs, env.n_uavs), dtype=bool)),
                        dtype=bool,
                    ).copy()
                )
                if not trajectory_record["follower_uav_ids"]:
                    meta = structured_obs.get("meta", {})
                    trajectory_record["follower_uav_ids"] = list(meta.get("follower_uav_ids", []))

            raw_obs = next_raw_obs

            if stop_on_termination and bool(torch.all(terminated_t | truncated_t).item()):
                ended_on_termination = True
                episode_end_visited_fractions.append(visited_fraction_t)
                if rollout_progress:
                    print(f"[collect_rollout] early stop on termination/truncation at step={steps_collected}")
                break

            if bool(torch.all(terminated_t | truncated_t).item()):
                episode_end_visited_fractions.append(visited_fraction_t)
                if reset_fn is not None:
                    raw_obs = reset_fn(env)
                else:
                    out = env.reset()
                    raw_obs = out[0] if isinstance(out, tuple) else out

        if ended_on_termination:
            last_values = torch.zeros_like(values)
        else:
            with torch.no_grad():
                last_structured_obs = self._to_device(obs_adapter(env, raw_obs), self.device)
                last_values = self.critic(**last_structured_obs["critic_obs"])

        self.buffer.compute_returns_and_advantages(
            last_values=last_values,
            normalize_advantages=self.config.normalize_advantages,
        )

        if rollout_progress:
            print(
                f"[collect_rollout] done: steps={steps_collected}, "
                f"ended_on_termination={ended_on_termination}"
            )

        rollout_sensor_fraction_visited_final = last_visited_fraction
        if len(episode_end_visited_fractions) > 0:
            rollout_episode_end_visited_fraction_mean = float(np.mean(episode_end_visited_fractions))
            rollout_episode_end_visited_fraction_last = float(episode_end_visited_fractions[-1])
        else:
            rollout_episode_end_visited_fraction_mean = float("nan")
            rollout_episode_end_visited_fraction_last = float("nan")

        result = {
            "rollout_reward_mean": total_reward / max(1, steps_collected),
            "rollout_reward_sum": total_reward,
            "rollout_reward_abs_mean": total_abs_reward / max(1, steps_collected),
            "rollout_reward_abs_sum": total_abs_reward,
            "rollout_nonzero_reward_step_frac": nonzero_reward_steps / max(1, steps_collected),
            "rollout_terminated_rate": total_terminated / max(1, steps_collected),
            "rollout_truncated_rate": total_truncated / max(1, steps_collected),
            "rollout_steps": float(steps_collected),
            "ended_on_termination": float(ended_on_termination),
            "rollout_sensor_fraction_visited_final": rollout_sensor_fraction_visited_final,
            "rollout_episode_end_visited_fraction_mean": rollout_episode_end_visited_fraction_mean,
            "rollout_episode_end_visited_fraction_last": rollout_episode_end_visited_fraction_last,
            "rollout_collision_penalty_sum": total_collision_penalty,
            "rollout_connectivity_penalty_sum": total_connectivity_penalty,
            "rollout_sensor_potential_sum": total_sensor_potential_reward,
            "rollout_sensor_collection_sum": total_sensor_collection_reward,
        }

        if trajectory_record is not None:
            trajectory_record["uav_locs"] = np.stack(trajectory_record["uav_locs"], axis=0)
            trajectory_record["times"] = np.asarray(trajectory_record["times"], dtype=np.float32)
            trajectory_record["reward_per_step"] = np.asarray(trajectory_record["reward_per_step"], dtype=np.float32)
            trajectory_record["visited_fraction"] = np.asarray(trajectory_record["visited_fraction"], dtype=np.float32)
            trajectory_record["visited_mask"] = np.stack(trajectory_record["visited_mask"], axis=0)
            trajectory_record["connectivity"] = np.stack(trajectory_record["connectivity"], axis=0)
            result["rollout_trajectory"] = trajectory_record

        return result

    def ppo_update(self) -> Dict[str, float]:
        self._check_required_actor_api()

        metrics = {
            "policy_loss": 0.0,
            "follower_policy_obj": 0.0,
            "entropy": 0.0,
            "value_loss": 0.0,
            "follower_approx_kl": 0.0,
            "updates": 0.0,
        }

        for _ in range(self.config.ppo_epochs):
            for mb in self.buffer.iter_minibatches(
                minibatch_size=self.config.minibatch_size,
                device=self.device,
                shuffle=True,
                drop_last=False,
            ):
                follower_obs = mb.follower_obs
                follower_mask = follower_obs.get("follower_mask")

                new_follower_log_prob, follower_entropy_matrix = self._forward_follower_eval(
                    follower_obs=follower_obs,
                    direction_action=mb.follower_direction_actions,
                )

                if follower_mask is not None:
                    follower_mask = follower_mask.bool()
                old_follower_log_prob = mb.follower_log_probs

                old_follower_joint_log_prob = self._masked_sum(old_follower_log_prob, follower_mask, dim=1)
                new_follower_joint_log_prob = self._masked_sum(new_follower_log_prob, follower_mask, dim=1)
                follower_entropy = self._masked_mean(follower_entropy_matrix, follower_mask, dim=1)

                advantages = mb.advantages

                follower_ratio = torch.exp(new_follower_joint_log_prob - old_follower_joint_log_prob)
                follower_surr1 = follower_ratio * advantages
                follower_surr2 = (
                    torch.clamp(follower_ratio, 1.0 - self.config.ppo_clip, 1.0 + self.config.ppo_clip) * advantages
                )
                follower_policy_obj = torch.min(follower_surr1, follower_surr2)

                policy_loss = -follower_policy_obj.mean() - self.config.entropy_coef * follower_entropy.mean()

                new_values = self.critic(**mb.critic_obs)
                value_loss = F.mse_loss(new_values, mb.returns)

                self.follower_actor_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.follower_actor.parameters(), self.config.grad_clip_norm)
                self.follower_actor_optimizer.step()

                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.config.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    follower_approx_kl = (old_follower_joint_log_prob - new_follower_joint_log_prob).mean()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["follower_policy_obj"] += float(follower_policy_obj.mean().item())
                metrics["entropy"] += float(follower_entropy.mean().item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["follower_approx_kl"] += float(follower_approx_kl.item())
                metrics["updates"] += 1.0

        if metrics["updates"] > 0:
            denom = metrics["updates"]
            for key in list(metrics.keys()):
                if key != "updates":
                    metrics[key] /= denom

        return metrics

    def train_ppo(
        self,
        env: Any,
        obs_adapter: Callable[[Any, Any], NestedData],
        action_adapter: Callable[[Dict[str, torch.Tensor], NestedData], Any],
        num_updates: int,
        reset_fn: Optional[Callable[[Any], Any]] = None,
        initial_obs: Optional[Any] = None,
        stop_on_termination: Optional[bool] = None,
        rollout_progress: Optional[bool] = None,
        rollout_progress_every: Optional[int] = None,
        record_rollout_trajectory: Optional[bool] = None,
        log_interval: int = 10,
        log_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        direction_temperature_initial: Optional[float] = None,
        direction_temperature_final: Optional[float] = None,
        temperature_schedule: str = "linear",
    ) -> list[Dict[str, Any]]:
        """
        Run MAPPO training for `num_updates` iterations.

        Each update performs:
          1) rollout collection into the internal buffer,
          2) PPO policy/value update on the collected rollout.
        """
        if num_updates <= 0:
            return []

        # Resolve initial direction temperature from the current config unless overridden.
        _dht_init = direction_temperature_initial if direction_temperature_initial is not None else self.config.direction_temperature
        _dht_final = direction_temperature_final if direction_temperature_final is not None else _dht_init
        _dht_schedule = direction_temperature_final is not None

        training_logs: list[Dict[str, Any]] = []
        current_initial_obs = initial_obs
        log_interval = max(1, int(log_interval))

        for update_idx in range(1, int(num_updates) + 1):
            # Apply direction temperature schedule before each rollout collection.
            if _dht_schedule:
                self.config.direction_temperature = _scheduled_temperature(
                    _dht_init, _dht_final, update_idx, num_updates, temperature_schedule
                )

            print(f"[train_ppo] collecting rollout {update_idx}/{num_updates}...")
            rollout_logs = self.collect_rollout(
                env=env,
                obs_adapter=obs_adapter,
                action_adapter=action_adapter,
                reset_fn=reset_fn,
                initial_obs=current_initial_obs,
                stop_on_termination=stop_on_termination,
                rollout_progress=rollout_progress,
                rollout_progress_every=rollout_progress_every,
                record_rollout_trajectory=record_rollout_trajectory,
            )
            print(
                f"[train_ppo] rollout {update_idx}/{num_updates} collected "
                f"(steps={int(rollout_logs.get('rollout_steps', 0))}, "
                f"reward_mean={float(rollout_logs.get('rollout_reward_mean', 0.0)):.4f})"
            )

            # Only the first update may consume an externally-provided initial_obs.
            # Later updates should follow the reset path inside collect_rollout.
            current_initial_obs = None

            update_logs = self.ppo_update()
            logs: Dict[str, Any] = {
                "update": float(update_idx),
                "direction_head_temperature": self.config.direction_temperature,
                **rollout_logs,
                **update_logs,
            }

            training_logs.append(logs)

            if log_callback is not None:
                log_callback(logs)

            if update_idx % log_interval == 0:
                print(
                    f"[train_ppo] update={update_idx}/{num_updates} "
                    f"rollout_reward_mean={logs.get('rollout_reward_mean', 0.0):.4f} "
                    f"rollout_reward_sum={logs.get('rollout_reward_sum', 0.0):.4f} "
                    f"rollout_nonzero_frac={logs.get('rollout_nonzero_reward_step_frac', 0.0):.4f} "
                    f"visited_final={logs.get('rollout_sensor_fraction_visited_final', 0.0):.4f} "
                    f"visited_ep_end_last={logs.get('rollout_episode_end_visited_fraction_last', float('nan')):.4f} "
                    f"policy_loss={logs.get('policy_loss', 0.0):.4f} "
                    f"value_loss={logs.get('value_loss', 0.0):.4f} "
                    f"follower_kl={logs.get('follower_approx_kl', 0.0):.6f}"
                )

        return training_logs

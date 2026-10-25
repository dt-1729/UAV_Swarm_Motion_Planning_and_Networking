# Copyright © MERL 2026
# Description: Stores and batches MAPPO rollout data for policy and value updates.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import torch

NestedData = Dict[str, Any]


def _to_tensor(data: Any) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.detach()
    return torch.as_tensor(data)


def _clone_to_cpu(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _clone_to_cpu(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_clone_to_cpu(v) for v in data)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    return torch.as_tensor(data).detach().cpu()


def _stack_nested(batch_list: List[Any]) -> Any:
    if len(batch_list) == 0:
        raise ValueError("Cannot stack an empty batch list")

    first = batch_list[0]
    if isinstance(first, dict):
        return {k: _stack_nested([entry[k] for entry in batch_list]) for k in first.keys()}

    tensors = [_to_tensor(x) for x in batch_list]
    return torch.stack(tensors, dim=0)


def _reshape_time_batch(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("Expected tensor with at least 2 dims [T, B, ...]")
    t, b = x.shape[0], x.shape[1]
    return x.reshape(t * b, *x.shape[2:])


def _flatten_time_batch_nested(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _flatten_time_batch_nested(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor) and data.ndim < 2:
        # Scalars (or already global constants) do not carry [T, B, ...] axes.
        # Keep them as-is; downstream code can broadcast as needed.
        return data
    return _reshape_time_batch(data)


def _index_nested(data: Any, idx: torch.Tensor) -> Any:
    if isinstance(data, dict):
        return {k: _index_nested(v, idx) for k, v in data.items()}
    if isinstance(data, torch.Tensor) and data.ndim == 0:
        return data
    return data[idx]


@dataclass
class RolloutBatch:
    follower_obs: NestedData
    critic_obs: NestedData
    follower_direction_actions: torch.Tensor
    follower_log_probs: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    extras: Optional[NestedData] = None


class MAPPORolloutBuffer:
    """
    On-policy rollout storage for follower-only PPO with centralized critic and GAE.

    Expected per-step shapes:
    - follower_direction_actions: (B, F, d)
    - follower_log_probs: (B, F)
    - rewards:      (B,)
    - terminated:   (B,) bool
    - truncated:    (B,) bool
    - values:       (B,)
    - nested observations can have any trailing dimensions after batch dimension B.
    """

    def __init__(self, rollout_length: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        if rollout_length <= 0:
            raise ValueError("rollout_length must be positive")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        if not (0.0 <= gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in [0, 1]")

        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self) -> None:
        self.follower_obs: List[NestedData] = []
        self.critic_obs: List[NestedData] = []
        self.follower_direction_actions: List[torch.Tensor] = []
        self.follower_log_probs: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.terminated: List[torch.Tensor] = []
        self.truncated: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        # V(next_obs) for terminal/truncated transitions, computed before any env reset.
        self.bootstrap_values: List[torch.Tensor] = []
        self.extras: List[NestedData] = []

        self._returns: Optional[torch.Tensor] = None
        self._advantages: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.rewards)

    @property
    def is_full(self) -> bool:
        return len(self) >= self.rollout_length

    def add(
        self,
        follower_obs: NestedData,
        critic_obs: NestedData,
        follower_direction_actions: Any,
        follower_log_probs: Any,
        rewards: Any,
        terminated: Any,
        truncated: Any,
        values: Any,
        bootstrap_values: Any = None,
        extras: Optional[NestedData] = None,
    ) -> None:
        if self.is_full:
            raise RuntimeError("Rollout buffer is full. Call reset() before adding new samples.")

        self.follower_obs.append(_clone_to_cpu(follower_obs))
        self.critic_obs.append(_clone_to_cpu(critic_obs))
        self.follower_direction_actions.append(_to_tensor(follower_direction_actions).detach().cpu())
        self.follower_log_probs.append(_to_tensor(follower_log_probs).detach().cpu())

        self.rewards.append(_to_tensor(rewards).cpu().float())
        self.terminated.append(_to_tensor(terminated).cpu().bool())
        self.truncated.append(_to_tensor(truncated).cpu().bool())
        values_t = _to_tensor(values).cpu().float()
        self.values.append(values_t)
        if bootstrap_values is None:
            bootstrap_values_t = torch.zeros_like(values_t)
        else:
            bootstrap_values_t = _to_tensor(bootstrap_values).cpu().float()
            if bootstrap_values_t.shape != values_t.shape:
                raise ValueError(
                    "bootstrap_values must have the same shape as values; "
                    f"got {tuple(bootstrap_values_t.shape)} vs {tuple(values_t.shape)}"
                )
        self.bootstrap_values.append(bootstrap_values_t)

        self.extras.append(_clone_to_cpu(extras if extras is not None else {}))

    def compute_returns_and_advantages(self, last_values: Any, normalize_advantages: bool = True) -> None:
        if len(self) == 0:
            raise RuntimeError("Cannot compute returns on an empty buffer.")

        rewards = _stack_nested(self.rewards).float()          # (T, B)
        terminated = _stack_nested(self.terminated).bool()     # (T, B)
        truncated = _stack_nested(self.truncated).bool()       # (T, B)
        values = _stack_nested(self.values).float()            # (T, B)
        bootstrap_values = _stack_nested(self.bootstrap_values).float()  # (T, B)
        last_values = _to_tensor(last_values).cpu().float()    # (B,)

        if last_values.ndim != 1:
            raise ValueError(f"last_values must have shape (B,), got {tuple(last_values.shape)}")

        t_steps, batch_size = rewards.shape
        if values.shape != (t_steps, batch_size):
            raise ValueError("values must have shape (T, B) matching rewards")
        if bootstrap_values.shape != (t_steps, batch_size):
            raise ValueError("bootstrap_values must have shape (T, B) matching rewards")

        done = (terminated | truncated).float()
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(batch_size, dtype=torch.float32)

        for t in reversed(range(t_steps)):
            not_done = 1.0 - done[t]
            not_terminated = (~terminated[t]).float()
            if t == t_steps - 1:
                next_values = torch.where(done[t].bool(), bootstrap_values[t], last_values)
            else:
                next_values = torch.where(done[t].bool(), bootstrap_values[t], values[t + 1])
            delta = rewards[t] + self.gamma * next_values * not_terminated - values[t]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae

        returns = advantages + values

        if normalize_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False).clamp_min(1e-8)
            advantages = (advantages - adv_mean) / adv_std

        self._returns = returns
        self._advantages = advantages

    def _stack_batch(self) -> RolloutBatch:
        if len(self) == 0:
            raise RuntimeError("Buffer is empty. Collect rollout data first.")
        if self._returns is None or self._advantages is None:
            raise RuntimeError("Returns/advantages are not computed. Call compute_returns_and_advantages().")

        return RolloutBatch(
            follower_obs=_stack_nested(self.follower_obs),
            critic_obs=_stack_nested(self.critic_obs),
            follower_direction_actions=_stack_nested(self.follower_direction_actions),
            follower_log_probs=_stack_nested(self.follower_log_probs),
            rewards=_stack_nested(self.rewards),
            terminated=_stack_nested(self.terminated),
            truncated=_stack_nested(self.truncated),
            values=_stack_nested(self.values),
            returns=self._returns,
            advantages=self._advantages,
            extras=_stack_nested(self.extras) if len(self.extras) > 0 else None,
        )

    def get_tensors(self, device: Optional[torch.device] = None) -> RolloutBatch:
        batch = self._stack_batch()
        if device is None:
            return batch

        def move(data: Any) -> Any:
            if isinstance(data, dict):
                return {k: move(v) for k, v in data.items()}
            if data is None:
                return None
            return data.to(device)

        return RolloutBatch(
            follower_obs=move(batch.follower_obs),
            critic_obs=move(batch.critic_obs),
            follower_direction_actions=move(batch.follower_direction_actions),
            follower_log_probs=move(batch.follower_log_probs),
            rewards=move(batch.rewards),
            terminated=move(batch.terminated),
            truncated=move(batch.truncated),
            values=move(batch.values),
            returns=move(batch.returns),
            advantages=move(batch.advantages),
            extras=move(batch.extras),
        )

    def iter_minibatches(
        self,
        minibatch_size: int,
        device: Optional[torch.device] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> Generator[RolloutBatch, None, None]:
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")

        batch = self.get_tensors(device=device)

        flat_follower_obs = _flatten_time_batch_nested(batch.follower_obs)
        flat_critic_obs = _flatten_time_batch_nested(batch.critic_obs)
        flat_follower_direction_actions = _reshape_time_batch(batch.follower_direction_actions)
        flat_follower_log_probs = _reshape_time_batch(batch.follower_log_probs)
        flat_rewards = _reshape_time_batch(batch.rewards)
        flat_terminated = _reshape_time_batch(batch.terminated)
        flat_truncated = _reshape_time_batch(batch.truncated)
        flat_values = _reshape_time_batch(batch.values)
        flat_returns = _reshape_time_batch(batch.returns)
        flat_advantages = _reshape_time_batch(batch.advantages)
        flat_extras = _flatten_time_batch_nested(batch.extras) if batch.extras is not None else None

        total_samples = flat_rewards.shape[0]
        indices = torch.randperm(total_samples, device=flat_rewards.device) if shuffle else torch.arange(
            total_samples, device=flat_rewards.device
        )

        for start in range(0, total_samples, minibatch_size):
            end = start + minibatch_size
            if end > total_samples and drop_last:
                break

            mb_idx = indices[start:end]
            yield RolloutBatch(
                follower_obs=_index_nested(flat_follower_obs, mb_idx),
                critic_obs=_index_nested(flat_critic_obs, mb_idx),
                follower_direction_actions=flat_follower_direction_actions[mb_idx],
                follower_log_probs=flat_follower_log_probs[mb_idx],
                rewards=flat_rewards[mb_idx],
                terminated=flat_terminated[mb_idx],
                truncated=flat_truncated[mb_idx],
                values=flat_values[mb_idx],
                returns=flat_returns[mb_idx],
                advantages=flat_advantages[mb_idx],
                extras=_index_nested(flat_extras, mb_idx) if flat_extras is not None else None,
            )

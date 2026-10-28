# Copyright © MERL 2026
# Description: Provides checkpointing, run directory, plotting, and TensorBoard helpers.
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import numbers
import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch

# TensorBoard only needs TensorFlow for optional features; an installed TensorFlow built against
# NumPy 1.x crashes on import under NumPy 2. Blocking it makes TensorBoard use its built-in stub.
sys.modules.setdefault("tensorflow", None)
from torch.utils.tensorboard import SummaryWriter


def format_float_tag(value: float, precision: int = 3) -> str:
    s = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def build_results_run_dir(
    env: Any,
    root_dir: str = "training_results",
    head_motion: Optional[str] = None,
) -> Path:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    head_tag = f"_hm_{head_motion}" if head_motion else ""
    dt_tag = format_float_tag(env.dt)
    sensor_potential_coeff = format_float_tag(env.potential_diff_coeff[1])
    run_name = (
        f"ns_{env.n_sensors}_nu_{env.n_uavs}"
        f"_dt_{dt_tag}_sp_{sensor_potential_coeff}{head_tag}_{run_stamp}"
    )
    run_dir = Path(root_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Lets the cluster job script move Slurm logs into the run directory.
    run_dir_file = os.environ.get("RUN_DIR_FILE")
    if run_dir_file:
        Path(run_dir_file).write_text(str(run_dir.resolve()) + "\n")
    return run_dir


def save_checkpoint(trainer: Any, save_path: Path) -> None:
    ckpt = {
        "follower_actor_state_dict": trainer.follower_actor.state_dict(),
        "critic_state_dict": trainer.critic.state_dict(),
        "follower_actor_optimizer_state_dict": trainer.follower_actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": trainer.critic_optimizer.state_dict(),
    }
    torch.save(ckpt, save_path)


def load_checkpoint(trainer: Any, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    trainer.follower_actor.load_state_dict(ckpt["follower_actor_state_dict"])
    trainer.critic.load_state_dict(ckpt["critic_state_dict"])

    if "follower_actor_optimizer_state_dict" in ckpt:
        trainer.follower_actor_optimizer.load_state_dict(ckpt["follower_actor_optimizer_state_dict"])
    if "critic_optimizer_state_dict" in ckpt:
        trainer.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])



def find_latest_checkpoint(base_dir: Path) -> Optional[Path]:
    candidates = sorted(base_dir.glob("**/model_final.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_latest_training_metrics(base_dir: Path) -> Optional[Path]:
    candidates = sorted(base_dir.glob("**/training_metrics.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def plot_training_results(
    df: pd.DataFrame,
    save_path: Path,
    smoothing_window: int = 1,
    show_variance: bool = False,
) -> None:
    smoothing_window = max(1, int(smoothing_window))

    plot_df = df.copy()
    if "update" in plot_df.columns:
        plot_df = plot_df.sort_values("update").reset_index(drop=True)

    cols_to_smooth = [
        "rollout_reward_mean",
        "rollout_sensor_fraction_visited_final",
        "policy_loss",
        "value_loss",
        "entropy",
        "follower_approx_kl",
        "rollout_terminated_rate",
        "rollout_truncated_rate",
    ]
    std_df = pd.DataFrame(index=plot_df.index)
    if smoothing_window > 1:
        for col in cols_to_smooth:
            if col in plot_df.columns:
                rolled = plot_df[col].rolling(window=smoothing_window, min_periods=1)
                std_df[col] = rolled.std().fillna(0.0)
                plot_df[col] = rolled.mean()
    else:
        for col in cols_to_smooth:
            if col in plot_df.columns:
                std_df[col] = 0.0

    def _plot_with_variance(ax_obj, x_vals, y_vals, col_name: str, label: str, linewidth: float = 1.8) -> None:
        ax_obj.plot(x_vals, y_vals, label=label, linewidth=linewidth)
        if show_variance and col_name in std_df.columns:
            std_vals = std_df[col_name]
            ax_obj.fill_between(x_vals, y_vals - std_vals, y_vals + std_vals, alpha=0.2)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ax = axes.ravel()

    x = plot_df["update"]

    _plot_with_variance(ax[0], x, plot_df["rollout_reward_mean"], "rollout_reward_mean", "rollout_reward_mean", linewidth=2.0)
    ax[0].set_title("Rollout Reward Mean")
    ax[0].set_xlabel("Update")
    ax[0].set_ylabel("Reward")
    ax[0].grid(alpha=0.3)

    _plot_with_variance(
        ax[1],
        x,
        plot_df["rollout_sensor_fraction_visited_final"],
        "rollout_sensor_fraction_visited_final",
        "rollout_sensor_fraction_visited_final",
        linewidth=2.0,
    )
    ax[1].set_title("Visited Fraction (Final)")
    ax[1].set_xlabel("Update")
    ax[1].set_ylabel("Fraction")
    ax[1].set_ylim(0.0, 1.0)
    ax[1].grid(alpha=0.3)

    _plot_with_variance(ax[2], x, plot_df["policy_loss"], "policy_loss", "policy_loss", linewidth=1.8)
    _plot_with_variance(ax[2], x, plot_df["value_loss"], "value_loss", "value_loss", linewidth=1.8)
    ax[2].set_title("Losses")
    ax[2].set_xlabel("Update")
    ax[2].set_ylabel("Loss")
    ax[2].legend()
    ax[2].grid(alpha=0.3)

    _plot_with_variance(ax[3], x, plot_df["entropy"], "entropy", "entropy", linewidth=2.0)
    ax[3].set_title("Entropy")
    ax[3].set_xlabel("Update")
    ax[3].set_ylabel("Entropy")
    ax[3].grid(alpha=0.3)

    _plot_with_variance(ax[4], x, plot_df["follower_approx_kl"], "follower_approx_kl", "follower_kl", linewidth=1.8)
    ax[4].set_title("Follower Approx KL")
    ax[4].set_xlabel("Update")
    ax[4].set_ylabel("KL")
    ax[4].legend()
    ax[4].grid(alpha=0.3)

    _plot_with_variance(ax[5], x, plot_df["rollout_terminated_rate"], "rollout_terminated_rate", "terminated", linewidth=1.8)
    _plot_with_variance(ax[5], x, plot_df["rollout_truncated_rate"], "rollout_truncated_rate", "truncated", linewidth=1.8)
    ax[5].set_title("Termination/Truncation")
    ax[5].set_xlabel("Update")
    ax[5].set_ylabel("Rate")
    ax[5].set_ylim(0.0, 1.0)
    ax[5].legend()
    ax[5].grid(alpha=0.3)

    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def build_tensorboard_logger(tb_dir: Path) -> Tuple[SummaryWriter, Callable[[Dict[str, Any]], None]]:
    writer = SummaryWriter(log_dir=str(tb_dir))

    def _tb_log_callback(logs: Dict[str, Any]) -> None:
        step = int(logs.get("update", 0))
        writer.add_scalar("rollout/reward_mean", float(logs.get("rollout_reward_mean", 0.0)), step)
        writer.add_scalar("rollout/visited_final", float(logs.get("rollout_sensor_fraction_visited_final", 0.0)), step)
        writer.add_scalar("rollout/terminated_rate", float(logs.get("rollout_terminated_rate", 0.0)), step)
        writer.add_scalar("rollout/truncated_rate", float(logs.get("rollout_truncated_rate", 0.0)), step)

        writer.add_scalar("loss/policy", float(logs.get("policy_loss", 0.0)), step)
        writer.add_scalar("loss/value", float(logs.get("value_loss", 0.0)), step)
        writer.add_scalar("entropy/total", float(logs.get("entropy", 0.0)), step)

        writer.add_scalar("kl/follower", float(logs.get("follower_approx_kl", 0.0)), step)

        # Log additional numeric update parameters/metrics automatically.
        # This keeps TensorBoard comprehensive as new keys are added.
        for key, value in logs.items():
            if key == "update":
                continue
            if isinstance(value, numbers.Real):
                writer.add_scalar(f"update/{key}", float(value), step)

        # Flush every update so values are visible live while training runs.
        writer.flush()

    return writer, _tb_log_callback

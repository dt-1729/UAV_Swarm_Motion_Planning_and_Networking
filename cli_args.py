# Copyright © MERL 2026
# Description: Defines command-line arguments for multi-agent PPO experiments.
import argparse
from typing import Optional, Sequence


_RUN_MODE_MAP = {
    "t": "train",
    "s": "sim",
    "e": "eval",
    "p": "plot",
}

_CBF_CONNECTIVITY_NEIGHBOR_MODE_MAP = {
    "u": "upstream",
    "n": "nearest",
}

_SENSOR_INIT_MODE_MAP = {
    "s": "stratified",
    "c": "clustered",
}


def parse_main_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI args for multi-agent main and normalize short string choices."""
    parser = argparse.ArgumentParser(description="Run MAPPO train/rollout/eval workflows.")
    parser.add_argument(
        "-r",
        "--run-mode",
        type=str,
        default="t",
        choices=["t", "s", "e", "p"],
        help="Execution mode: t=train, s=sim (random follower model), e=eval, p=plot.",
    )
    parser.add_argument(
        "-ec",
        "--eval-checkpoint-path",
        type=str,
        default=None,
        help="Trained checkpoint path for sim and eval modes (sim uses a random model if omitted).",
    )
    parser.add_argument(
        "-cu",
        "--checkpoint-every-updates",
        type=int,
        default=25,
        help="Save a checkpoint every N training updates (<=0 disables periodic checkpointing).",
    )
    parser.add_argument(
        "-pc",
        "--plot-csv-path",
        type=str,
        default=None,
        help="Path to training_metrics.csv for post-training plotting (plot mode).",
    )
    parser.add_argument(
        "-po",
        "--plot-output-path",
        type=str,
        default=None,
        help="Output PNG path for post-training plotting (plot mode).",
    )
    parser.add_argument(
        "-sw",
        "--smoothing-window",
        type=int,
        default=10,
        help="Moving-average window for training plots (>=1).",
    )
    parser.add_argument(
        "-sv",
        "--show-variance",
        action="store_true",
        help="Show rolling variance bands (mean +/- std) in training plots.",
    )
    parser.add_argument(
        "-af",
        "--anim-frame-stride",
        type=int,
        default=5,
        help="Frame subsampling stride for rollout/eval GIF animation (>=1). Larger is faster.",
    )
    parser.add_argument(
        "-ai",
        "--anim-interval-ms",
        type=int,
        default=100,
        help="Frame interval in milliseconds for rollout/eval GIF animation (>=10).",
    )
    parser.add_argument(
        "-rl",
        "--rollout-steps",
        type=int,
        default=6000,
        help="Rollout length in steps for MAPPO collection.",
    )
    parser.add_argument(
        "-mt",
        "--max-time",
        type=float,
        default=200.0,
        help="Environment episode time horizon.",
    )
    parser.add_argument(
        "-ep",
        "--epochs",
        type=int,
        default=200,
        help="Number of training updates (epochs).",
    )
    parser.add_argument(
        "-mb",
        "--minibatch-size",
        type=int,
        default=256,
        help="Minibatch size for PPO updates.",
    )
    parser.add_argument(
        "-fc",
        "--finetune-checkpoint-path",
        type=str,
        default=None,
        help="Optional checkpoint to load weights from before training (finetune/resume).",
    )
    parser.add_argument(
        "-dht",
        "--direction-head-temperature",
        type=float,
        default=1.0,
        help="Temperature for the follower direction-head distribution (>0). Lower values make the distribution sharper.",
    )
    parser.add_argument(
        "-dhtf",
        "--direction-head-temperature-final",
        type=float,
        default=None,
        help="Final direction-head temperature for scheduling. If None, no scheduling (constant at --direction-head-temperature).",
    )
    parser.add_argument(
        "-ts",
        "--temperature-schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine", "exponential"],
        help="Schedule type for temperature annealing during training: linear, cosine, or exponential.",
    )
    parser.add_argument(
        "-umd",
        "--use-mean-direction",
        action="store_true",
        help="Use vmf3 mean direction (mu) instead of sampling for environment steps.",
    )
    parser.add_argument(
        "-cnm",
        "--cbf-connectivity-neighbor-mode",
        type=str,
        default="u",
        choices=["u", "n"],
        help="CBF connectivity neighbor selection: u=upstream, n=nearest.",
    )
    parser.add_argument(
        "-nu",
        "--n-uavs",
        type=int,
        default=3,
        help="Number of UAVs in the environment, including the head (>=2).",
    )
    parser.add_argument(
        "-ns",
        "--n-sensors",
        type=int,
        default=10,
        help="Number of sensors in the environment (>=1).",
    )
    parser.add_argument(
        "-sim",
        "--sensor-init-mode",
        type=str,
        default="s",
        choices=["s", "c"],
        help="Sensor initialization mode: s=stratified, c=clustered.",
    )
    parser.add_argument(
        "-nsc",
        "--n-sensor-clusters",
        type=int,
        default=3,
        help="Number of spatial sensor clusters when --sensor-init-mode=c (>=1).",
    )
    parser.add_argument(
        "-hm",
        "--head-motion",
        type=str,
        default="c",
        choices=["s", "r", "c"],
        help="Head UAV motion: s=stationary, r=smooth random trajectory, c=cluster TSP tour over sensors.",
    )
    parser.add_argument(
        "-hnc",
        "--head-n-clusters",
        type=int,
        default=5,
        help="Number of sensor clusters the head visits in --head-motion=c.",
    )
    parser.add_argument(
        "-nc",
        "--n-chains",
        type=int,
        default=2,
        help="Number of follower chains rooted at head UAV (>=1).",
    )

    args = parser.parse_args(argv)

    # Keep CLI compact while normalizing to canonical names used in training logic.
    args.run_mode = _RUN_MODE_MAP[args.run_mode]
    args.cbf_connectivity_neighbor_mode = _CBF_CONNECTIVITY_NEIGHBOR_MODE_MAP[
        args.cbf_connectivity_neighbor_mode
    ]
    args.sensor_init_mode = _SENSOR_INIT_MODE_MAP[args.sensor_init_mode]
    args.head_motion = {"s": "stationary", "r": "random", "c": "cluster_tsp"}[args.head_motion]
    args.head_n_clusters = max(1, int(args.head_n_clusters))

    # Normalize numeric ranges once so main code stays clean.
    args.smoothing_window = max(1, int(args.smoothing_window))
    args.anim_frame_stride = max(1, int(args.anim_frame_stride))
    args.anim_interval_ms = max(10, int(args.anim_interval_ms))
    args.rollout_steps = max(1, int(args.rollout_steps))
    args.epochs = max(1, int(args.epochs))
    args.minibatch_size = max(1, int(args.minibatch_size))
    args.checkpoint_every_updates = int(args.checkpoint_every_updates)
    args.max_time = float(args.max_time)
    args.direction_head_temperature = float(args.direction_head_temperature)
    args.n_uavs = max(2, int(args.n_uavs))
    args.n_sensors = max(1, int(args.n_sensors))
    args.n_sensor_clusters = max(1, int(args.n_sensor_clusters))
    args.n_chains = max(1, int(args.n_chains))
    if args.direction_head_temperature_final is not None:
        args.direction_head_temperature_final = float(args.direction_head_temperature_final)

    return args
